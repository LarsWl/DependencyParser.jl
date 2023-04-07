export train!

import .ArcEager: set_labels!, execute_transition, zero_cost_transitions, transition_costs, optimal_transition_index, is_transition_valid, gold_scores
import .ArcEager: GoldState, FORBIDDEN_COST

using Flux
using StatsBase
using ProgressMeter: @showprogress
using Dates
using CUDA

function train_predict_tree(model::Model, sentence::Sentence, context::TrainingContext)
	config = Configuration(sentence)
	transitions_number = 0

	while !is_terminal(config)
		transition = predict_transition(model, context.settings, context.system, config)
		transition === nothing && break
		execute_transition(config, transition, context.system)

		transitions_number += 1
		transitions_number > LIMIT_TRANSITIONS_NUMBER && break
	end

	config.tree
end

log(message) = @info "$(Dates.now()) $message"

function train!()
	model, training_context, dataset = init_train()
	
	train_samples = Flux.DataLoader(dataset, batchsize=training_context.settings.sample_size, shuffle=true)
	test_sample = sample(dataset, 100; replace=false)
	# test_gold= cu(hcat(map(last, test_sample)...))
	test_gold= hcat(map(last, test_sample)...)
	test_features = map(first, test_sample)

	training_context.test_dataset = build_dataset(model, sample(training_context.test_connlu_sentences, 50; replace=false), training_context)
	
	train_epoch = () -> begin
		count = 0

		for train_sample in train_samples
			if count % 5 == 0 || count == 1
				@time update_model!(model, train_sample, training_context)
			else
				update_model!(model, train_sample, training_context)
			end
			
			GC.gc()

			pre_map = init_pre_compute_map(test_sample, training_context)
			saved = pre_compute(pre_map, model, training_context)

			if count % 3 == 0
				log("Current loss: $(transitions_loss(model, test_gold, test_features, training_context, pre_map, saved))")
			end
 
			if count % 10 == 0
				@info "Start calculating UAS/LAS..."
				@time test_training_scores(model, training_context)
			end

			count += 1
		end

		@info "Start calculating UAS/LAS..."
		test_training_scores(model, training_context)
	end

	@info "Start training"
	Flux.@epochs 500 train_epoch()
end

function init_train()
	log("Initializing model...")

	system = ArcEager.ArcEagerSystem()
	train_file = "/home/egor/UD_English-EWT/en_ewt-ud-train.conllu"
	test_file = "/home/egor/UD_English-EWT/en_ewt-ud-dev.conllu"
	model_file = "tmp/model_v1"
	results_file = "tmp/results_v1"

	connlu_sentences = load_connlu_file(train_file)
	settings = Settings(embeddings_size=100)
	model = Model(settings, system, connlu_sentences)
	# enable_cuda(model)

	Flux.testmode!(model.dropout, false)

	sort(collect(model.label_ids), by=pair->pair[end]) |>
		pairs -> map(pair -> pair[begin], pairs) |>
		labels -> set_labels!(system, labels)

	test_sentences = load_connlu_file(test_file)

	training_context = TrainingContext(
		system,
		settings,
		train_file,
		connlu_sentences,
		test_sentences,
		test_file,
		results_file,
		model_file,
		beam_coef = 0.05
	)

	training_context.optimizer = ADAM()

	dataset = init_dataset()

	(model, training_context, dataset)
end

# global cnt = 0
# global cnt2 = 0

# test_sample = sample(dataset, 2_000; replace=false)

# test_sample[begin][begin]

# @time update_model!(model, test_sample, training_context)
# @show cnt
# @show cnt2
# transitions_loss(model, test_gold, test_features, training_context, pre_map, saved)

# scores = test_features .|> feature -> calculate_scores(model, feature, training_context, pre_map, saved)
# test_gold

function update_model!(model::Model, train_sample, training_context::TrainingContext)
	ps = Flux.params(model)

	pre_map = init_pre_compute_map(train_sample, training_context)
	saved = pre_compute(pre_map, model, training_context)

	@show length(pre_map)
	@show size(saved)

	# gold = cu(hcat(map(last, train_sample)...))
	gold = hcat(map(last, train_sample)...)
	features = map(first, train_sample)

	grads = gradient(ps) do
		transitions_loss(model, gold, features, training_context, pre_map, saved) +
			L2_norm(ps, training_context.settings)
	end

	Flux.update!(training_context.optimizer, ps, grads)
end

function transitions_loss(model, gold, features, training_context, pre_map, saved)
	scores = features .|> feature -> calculate_scores(model, feature, training_context, pre_map, saved)

	Flux.crossentropy(hcat(scores...), gold)
end

function calculate_scores(model, feature, training_context, pre_map, saved)
	# hidden = cu(zeros(Float32, training_context.settings.hidden_size))
	hidden = zeros(Float32, training_context.settings.hidden_size)

	for i in 1:training_context.settings.batch_size
		tok = feature[i]
		index = tok * training_context.settings.batch_size + i - 1

		if haskey(pre_map, index)
			global cnt2 += 1
			id = pre_map[index]

			hidden += saved[id]
		else
			global cnt += 1

			offset = (i - 1) * training_context.settings.embeddings_size
			hidden_slice = view(model.hidden_layer.weight, :, (offset + 1):(offset + training_context.settings.embeddings_size))

			hidden += hidden_slice * view(model.embeddings, tok, :)
		end
	end
	
	hidden += model.hidden_layer.bias

	(hidden .^ 3) |> model.dropout |> model.output_layer |> softmax
end

function init_pre_compute_map(dataset, training_context)
	counter = Dict{Integer, Integer}()
	foreach(dataset) do example
		foreach(enumerate(first(example))) do (i, f)
			key = f * training_context.settings.batch_size + i - 1

			if haskey(counter, key)
				counter[key] += 1
			else
				counter[key] = 1
			end
		end
	end

	num_precomputed = min(length(counter), 10_000)

	to_precompute = sort(collect(counter); by=last,rev=true)[begin:begin + num_precomputed - 1] .|> first

	map(pair -> (last(pair), first(pair)), enumerate(to_precompute)) |> Dict
end

function pre_compute(pre_map, model, training_context)
	saved = zeros(length(pre_map), training_context.settings.hidden_size)

	hidden_weight = collect(model.hidden_layer.weight)
	embeddings = collect(model.embeddings)

	for index in keys(pre_map)
		map_X = pre_map[index]
		tok = div(index, training_context.settings.batch_size)
		pos = mod(index, training_context.settings.batch_size)

		saved[map_X, :] = matrix_multiply_slice_sum(
			saved[map_X, :],
			hidden_weight,
			embeddings[tok, :],
			pos * training_context.settings.embeddings_size
		)
	end

	map(1:first(size(saved))) do i
		# cu(saved[i, :])
		saved[i, :]
	end
end

function matrix_multiply_slice_sum(sum, matrix, vector, left_column_offset)
  for i in 1:first(size(matrix))
    partial = sum[i]

    for j in 1:length(vector)
      partial += matrix[i, left_column_offset + j] * vector[j]
    end

    sum[i] = partial;
  end

  sum
end

function L2_norm(ps, settings::Settings)
	sqnorm(x) = sum(abs2, x)

	sum(sqnorm, ps) * (settings.reg_weight / 2)
end

function build_dataset(model, sentences_batch, training_context)
	train_samples = []
	mutex = ReentrantLock()
	sent_mutex = ReentrantLock()
	sentences_number = 0

	parse_gold_tree = (connlu_sentence) -> begin
		batch = []

		sentence = zip(connlu_sentence.token_doc.tokens, connlu_sentence.pos_tags) |> collect |> Sentence

		config = Configuration(sentence)
		process_config(model, config, connlu_sentence.gold_tree, training_context, 0, mutex, batch)

		lock(sent_mutex) do
			train_samples = vcat(train_samples, batch)
			sentences_number += 1
			if sentences_number % 250 == 0
				@info "Sentences processed: $sentences_number"
			end
		end
	end


	try
		Threads.@sync begin
			@info "Build dataset..."
			Threads.@threads for connlu_sentence in sentences_batch
				parse_gold_tree(connlu_sentence)
			end
		end
	catch err
		println(err.task.exception)
		return err
	end

	@info "Dataset builded..."

	train_samples
end

function process_config(model, config, gold_tree, context, transition_number, mutex, train_samples)
	is_terminal(config) && return
	transition_number >= LIMIT_TRANSITIONS_NUMBER && return
  
	gold_state = GoldState(gold_tree, config, context.system)
	zero_transitions = zero_cost_transitions(gold_state)
	length(zero_transitions) == 0 && return

	gold  = transition_costs(gold_state) |> gold_scores
  
	batch = form_batch(model, context.settings, config)
  
	lock(mutex) do 
	  push!(train_samples, (batch, gold))
	end
  
	transition = context.system.transitions[Int8(gold.indices)]
	execute_transition(config, transition, context.system)

	process_config(model, config, gold_tree, context, transition_number + 1, mutex, train_samples)
end

function test_training_scores(model::Model, context::TrainingContext)
	losses = []
	parsed_trees_file = "tmp/parsed_trees.txt"
	mutex = ReentrantLock()
	losses_mutex = ReentrantLock()
	parse_tree = (connlu_sentence, file) -> begin
		sentence = zip(connlu_sentence.token_doc.tokens, connlu_sentence.pos_tags) |> collect |> Sentence

		tree = train_predict_tree(model, sentence, context)
		tree_text = convert_to_string(tree)

		lock(mutex) do
			write(file, tree_text)
			write(file, "\n")
			write(file, "\n")
		end
	end

	parse_sample = (sample) -> begin
		loss = test_loss(model, sample, context)

		lock(losses_mutex) do
			push!(losses, loss)
		end
	end

	open(parsed_trees_file, "w") do file
		for i in 1:length(context.test_connlu_sentences)
			connlu_sentence = context.test_connlu_sentences[i]

			parse_tree(connlu_sentence, file)
		end
	end

	parse_sample(context.test_dataset)

	conllu_source = DependencyParserTest.Sources.ConlluSource(context.test_connlu_file)
	parser_source = DependencyParserTest.Sources.CoreNLPSource(parsed_trees_file)

	uas, las = DependencyParserTest.Benchmark.test_accuracy(conllu_source, parser_source)
	best_loss = min(losses...)
	worst_loss = max(losses...)
	avg_loss = sum(losses) / length(losses)

	open(context.training_results_file, "a") do file
		result_line = "UAS=$(uas), LAS=$(las), best_loss=$(best_loss), worst_loss=$(worst_loss), avg_loss=$(avg_loss)\n"
		write(file, result_line)
	end

	save(model, context.model_file * "_last.bson")

	if uas > context.best_uas || (uas == context.best_uas && las > context.best_las)
		context.best_uas = uas
		context.best_las = las
		context.best_loss = avg_loss

		save(model, context.model_file * "_best.bson")
	end
end

function test_loss(model::Model, sample, context)
	ps = Flux.params(model)

	sum(sample) do (batch, gold)
		predict(model, batch) |> scores -> test_transition_loss(scores, gold)
	end + L2_norm(ps, context.settings)
end

function test_transition_loss(scores, gold)
	Flux.Losses.crossentropy(scores, gold)
end
