using Flux
using StatsBase

const grad_saved_mutex = ReentrantLock()

mutable struct CostAndGrad
	cost
	correct
	grad_w1
	grad_b1
	grad_w2
	grad_e
end

function train_corenlp!()
	@info "Initializing model..."

	system = ArcEager.ArcEagerSystem()
	train_file = "/home/egor/UD_English-EWT/en_ewt-ud-train.conllu"
	test_file = "/home/egor/UD_English-EWT/en_ewt-ud-dev.conllu"
	model_file = "tmp/model_v1.bson"
	results_file = "tmp/results_v1"

	connlu_sentences = load_connlu_file(train_file)
	settings = Settings(embeddings_size=100)
	model = Model(settings, system, connlu_sentences)

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

	@info "Initialazing dataset..."

	dataset = init_dataset()
	examples_batches = Flux.DataLoader(dataset, batchsize=training_context.examples_batch_size, shuffle=true)
	opt = ADAM()

	for examples_batch in examples_batches
		@info "Start process batch with $(training_context.examples_batch_size) examples"

		cost_and_grad = @time calc_cost_and_gradients(examples_batch, model, training_context)

		@info "Cost: $(cost_and_grad.cost)"
		@info "Percent correct: $(cost_and_grad.correct / training_context.examples_batch_size)"

		ps = Flux.params(model)
		gs = model_grads(model, cost_and_grad)

		Flux.update!(opt, ps, gs)

		save(model, model_file)
	end
end

function merge!(dest_cost::CostAndGrad, src_cost::CostAndGrad)
	dest_cost.cost += src_cost.cost
	dest_cost.correct += src_cost.correct

	dest_cost.grad_w1 += src_cost.grad_w1
	dest_cost.grad_b1 += src_cost.grad_b1
	dest_cost.grad_w2 += src_cost.grad_w2
	dest_cost.grad_e += src_cost.grad_e
end

function model_grads(model, cost_and_grad::CostAndGrad)
	IdDict(
		model.embeddings => cost_and_grad.grad_e,
		model.hidden_layer.weight => cost_and_grad.grad_w1,
		model.hidden_layer.bias => cost_and_grad.grad_b1,
		model.output_layer.weight => cost_and_grad.grad_w2
	)
end

function calc_cost_and_gradients(examples_batch, model, training_context)
	pre_map = init_pre_compute_map(examples_batch, training_context)
	saved = pre_compute(pre_map, model, training_context)
	grad_saved = zeros(size(saved))

	num_chunks = Threads.nthreads()
	chunks = Flux.DataLoader(examples_batch, batchsize = Integer(ceil(length(examples_batch) / num_chunks)))

	@info "Number of chunks: $num_chunks"
	@info "Pre computed: $(length(pre_map))"

	final_cost = nothing
	final_cost_mutex = ReentrantLock()

	for chunk in chunks
		part_cost = part_calc_cost_and_gradients(chunk, model, training_context, pre_map, saved, grad_saved)

		lock(final_cost_mutex) do
			if final_cost === nothing
				final_cost = part_cost
			else
				merge!(final_cost, part_cost)
			end
		end
	end

	@info "Backprop save and add regulariztion..."
	backprop_saved(final_cost, model, pre_map, grad_saved, training_context)
	add_L2_regularization(model, final_cost, training_context.settings.reg_weight)

	final_cost
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

	num_precomputed = min(length(counter), 20_000)

	to_precompute = sort(collect(counter),by=last,rev=true)[begin:begin + num_precomputed - 1] .|> first

	map(pair -> (last(pair), first(pair)), enumerate(to_precompute)) |> Dict
end

function pre_compute(pre_map, model, training_context)
	saved = zeros(length(pre_map), training_context.settings.hidden_size)

	for index in keys(pre_map)
		map_X = pre_map[index]
		tok = div(index, training_context.settings.batch_size)
		pos = mod(index, training_context.settings.batch_size)

		saved[map_X, :] = matrix_multiply_slice_sum(
			saved[map_X, :],
			model.hidden_layer.weight,
			model.embeddings[tok, :],
			pos * training_context.settings.embeddings_size
		)
	end

	saved
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

function part_calc_cost_and_gradients(examples, model, training_context, pre_map, saved, grad_saved)
	grad_w1 = zeros(size(model.hidden_layer.weight))
	grad_b1 = zeros(size(model.hidden_layer.bias))
	grad_w2 = zeros(size(model.output_layer.weight))
	grad_e = zeros(size(model.embeddings))

	cost = 0.0;
	correct = 0.0;
	batch_size = training_context.examples_batch_size
	processed = 0

	for example in examples
		feature = first(example)
		label = last(example)

		scores = zeros(Float32, transitions_number(training_context.system))
		hidden = zeros(Float32, training_context.settings.hidden_size)
		hidden3 = zeros(Float32, training_context.settings.hidden_size)

		ls = filter(>(0), Flux.dropout(collect(1:training_context.settings.hidden_size), 0.5, dims=1) ./ 2 .|> Int32)

		offset = 0
		for j in 1:training_context.settings.batch_size
			tok = feature[j]
			index = tok * training_context.settings.batch_size + j

			if haskey(pre_map, index)
				id = pre_map[index]

				for node_index in ls 
					hidden[node_index] += saved[id, node_index]
				end
			else
				for node_index in ls 
					for k in 1:training_context.settings.embeddings_size
						hidden[node_index] += model.hidden_layer.weight[node_index, offset + k] * model.embeddings[tok, k]
					end
				end
				offset += training_context.settings.embeddings_size
			end
		end

		# Add bias term and apply activation function
		for node_index in ls
			hidden[node_index] += model.hidden_layer.bias[node_index];
			hidden3[node_index] = hidden[node_index] ^ 3
		end

		# Feed forward to softmax layer (no activation yet)
		opt_label = -1
		for i in 1:transitions_number(training_context.system)
			if label[i] >= 0
				for node_index in ls
					scores[i] += model.output_layer.weight[i, node_index] * hidden3[node_index];
				end

				if (opt_label < 0 || scores[i] > scores[opt_label])
					opt_label = i;
				end
			end
		end

		sum1 = 0.0
		sum2 = 0.0
		max_score = scores[opt_label]
		for i in 1:transitions_number(training_context.system)
			if label[i] >= 0
				scores[i] = exp(scores[i] - max_score)

				if label[i] == 1
					sum1 += scores[i]
				end

				sum2 += scores[i]
			end
		end

		cost += (log(sum2) - log(sum1)) / batch_size
		if label[opt_label] == 1
			correct += +1.0 / batch_size;
		end

		grad_hidden3 = zeros(Float32, training_context.settings.hidden_size)
		for i in 1:transitions_number(training_context.system)
			if label[i] >= 0
				delta = -(label[i] - scores[i] / sum2) / batch_size;

				for node_index in ls
					grad_w2[i, node_index] += delta * hidden3[node_index];
					grad_hidden3[node_index] += delta * model.output_layer.weight[i, node_index];
				end
			end
		end

		grad_hidden = zeros(Float32, training_context.settings.hidden_size)
		for node_index in ls
			grad_hidden[node_index] = grad_hidden3[node_index] * 3 * hidden[node_index] * hidden[node_index];
			grad_b1[node_index] += grad_hidden[node_index];
		end

		offset = 0
		for j in 1:training_context.settings.batch_size
			tok = feature[j]
			index = tok * training_context.settings.batch_size + j

			if haskey(pre_map, index)
				id = pre_map[index]

				lock(grad_saved_mutex) do
					for node_index in ls
						grad_saved[id, node_index] += grad_hidden[node_index]
					end
				end
			else
				for node_index in ls
					for k in 1:training_context.settings.embeddings_size
						grad_w1[node_index, offset + k] += grad_hidden[node_index] * model.embeddings[tok, k]
						grad_e[tok, k] += grad_hidden[node_index] * model.hidden_layer.weight[node_index, offset + k]
					end
				end
			
				offset += training_context.settings.embeddings_size
			end
		end

		processed += 1

		if processed % 400 == 0
			@info "400 examples processed..."
		end
	end

	CostAndGrad(cost, correct, grad_w1, grad_b1, grad_w2, grad_e)
end

function add_L2_regularization(model, cost, reg_weight)
	for i in 1:first(size(model.hidden_layer.weight))
		for j in 1:first(last(model.hidden_layer.weight))
			cost.cost += reg_weight * model.hidden_layer.weight[i, j] * model.hidden_layer.weight[i, j] / 2.0
			cost.grad_w1[i, j] += reg_weight * model.hidden_layer.weight[i, j]
		end
	end

	for i in 1:length(model.hidden_layer.bias)
		cost.cost += reg_weight * model.hidden_layer.bias[i] * model.hidden_layer.bias[i] / 2.0
		cost.grad_b1[i] += reg_weight * model.hidden_layer.bias[i]
	end

	for i in 1:first(size(model.output_layer.weight))
		for j in 1:first(last(model.output_layer.weight))
			cost.cost += reg_weight * model.output_layer.weight[i, j] * model.output_layer.weight[i, j] / 2.0
			cost.grad_w2[i, j] += reg_weight * model.output_layer.weight[i, j]
		end
	end


	for i in 1:first(size(model.embeddings))
		for j in 1:first(last(model.embeddings))
			cost.cost += reg_weight * model.embeddings[i, j] * model.embeddings[i, j] / 2.0
			cost.grad_e[i, j] += reg_weight * model.embeddings[i, j]
		end
	end
end

function backprop_saved(final_cost, model, pre_map, grad_saved, training_context)
	for index in keys(pre_map)
		map_X = pre_map[index]
		tok = div(index, training_context.settings.batch_size)
		offset = (index % training_context.settings.batch_size) * training_context.settings.embeddings_size;
		for j in 1:training_context.settings.hidden_size
			delta = grad_saved[map_X, j]

			for k in 1:training_context.settings.embeddings_size
				final_cost.grad_w1[j, offset + k] += delta * model.embeddings[tok, k]
				final_cost.grad_e[tok, k] += delta * model.hidden_layer.weight[j, offset + k];
			end
		end
	end
end
