using DependencyParser.DependencyParsing

@testset "Model" begin
  @testset "initialize untrained with embeddings" begin
    connlu_file = "test/fixtures/connlu_test.txt"
    embeddings_file = "test/fixtures/embeddings_test.vec"
    connlu_sentences = load_connlu_file(connlu_file)
    system = ArcEager.ArcEagerSystem()
    settings = Settings(embeddings_size = 4, hidden_size = 2, batch_size = 3)

  model = Model(settings, system, embeddings_file, connlu_sentences)
    @test length(model.embeddings[1, :]) == settings.embeddings_size
    @test length(model.hidden_layer.weight[:, 1]) == settings.hidden_size
    @test length(model.hidden_layer.weight[1, :]) == settings.batch_size * settings.embeddings_size
    @test length(model.output_layer.weight[:, 1]) == ArcEager.transitions_number(system)
  end
  
  @testset "Write to file" begin
    filename = "tmp/modelfile.txt"
    settings = Settings(embeddings_size = 4, hidden_size = 2, batch_size = 3)
    labels_num = 10
    model = build_model(settings)

    words_count = length(model.word_ids)
    tags_count = length(model.tag_ids)
    labels_count = length(model.label_ids)
    total_ids_count = words_count + tags_count + labels_count

    total_lines_count = total_ids_count + settings.hidden_size + labels_num + 3

    write_to_file!(model, filename)

    word = first(model.word_ids)
    tag = first(model.tag_ids)
    label = first(model.label_ids)

    # manualy count from model and settings
    correct_embeddings_info_line = "6 6 3 4"
    correct_layers_info_line = "3 2 10"

    word_correct_line = "$(word[begin]) $(join(model.embeddings[word[end], :], " "))"
    tag_correct_line = "$(tag[begin]) $(join(model.embeddings[tag[end], :], " "))"
    label_correct_line = "$(label[begin]) $(join(model.embeddings[label[end], :], " "))"

    hidden_weight_correct_line = join(model.hidden_layer.weight[begin, :], " ")
    hidden_bias_correct_line = join(model.hidden_layer.bias, " ")
    softmax_weight_correct_line = join(model.output_layer.weight[begin, :], " ")

    lines = readlines(filename)

    @test length(lines) == total_lines_count
    @test lines[begin] == correct_embeddings_info_line
    @test lines[begin + 1] == word_correct_line
    @test lines[begin + length(model.word_ids) + 1] == tag_correct_line
    @test lines[begin + length(model.word_ids) + length(model.tag_ids) + 1] == label_correct_line
    @test lines[begin + length(model.word_ids) + length(model.tag_ids) + length(model.label_ids) + 1] == correct_layers_info_line
    @test lines[begin + length(model.word_ids) + length(model.tag_ids) + length(model.label_ids) + 2] == hidden_weight_correct_line
    @test lines[begin + length(model.word_ids) + length(model.tag_ids) + length(model.label_ids) + settings.hidden_size + 2] == hidden_bias_correct_line
    @test lines[begin + length(model.word_ids) + length(model.tag_ids) + length(model.label_ids) + settings.hidden_size + 3] == softmax_weight_correct_line

    rm(filename)
  end

  @testset "Read from file" begin
    filename = "test/fixtures/modelfile_test.txt"

    model = Model(filename)

    # manualy calculated from fixture model
    word_pair = Pair{String, Integer}("here", 1)
    tag_pair = Pair{String, Integer}("TAG2", 7)
    label_pair = Pair{String, Integer}("LABEL2", 13)
    word_embedding = Float32[1, 2, 3, 4]
    tag_embedding = Float32[3, 2, 4, 1]
    label_embedding = Float32[4, 2, 1, 3]
    hidden_size = 2
    batch_size = 3
    labels_num = 10
    hidden_layer_weight_vector = Float32[1,2,3,4,5,6,7,8,9,10,9,8]
    hidden_layer_bias = Float32[1, 2]
    softmax_layer_weight_vector = [3, 4]

    @test length(model.word_ids) == 6
    @test length(model.tag_ids) == 6
    @test length(model.label_ids) == 3
    @test haskey(model.word_ids, word_pair[begin]) && model.word_ids[word_pair[begin]] == word_pair[end]
    @test haskey(model.tag_ids, tag_pair[begin]) && model.tag_ids[tag_pair[begin]] == tag_pair[end]
    @test haskey(model.label_ids, label_pair[begin]) && model.label_ids[label_pair[begin]] == label_pair[end]
    @test model.embeddings[word_pair[end], :] == word_embedding
    @test model.embeddings[tag_pair[end], :] == tag_embedding
    @test model.embeddings[label_pair[end], :] == label_embedding
    @test length(model.hidden_layer.bias) == hidden_size
    @test length(model.output_layer.weight[begin, :]) == hidden_size
    @test length(model.output_layer.weight[:, begin]) == labels_num
    @test model.hidden_layer.bias == hidden_layer_bias
    @test model.hidden_layer.weight[begin, :] == hidden_layer_weight_vector
    @test model.output_layer.weight[begin, :] == softmax_layer_weight_vector
  end

  @testset "Model prediction" begin
    parser = build_dep_parser()
    config = build_configuration()
    model = parser.model
    batch = build_correct_batch(parser, config)

    prediction = predict(model, batch)

    @test length(prediction) == ArcEager.transitions_number(parser.parsing_system)
    @test round(sum(prediction)) == 1
  end
end