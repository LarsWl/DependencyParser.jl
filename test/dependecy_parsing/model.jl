using DependencyParser.DependencyParsing

@testset "Model" begin
  connlu_file = "test/fixtures/connlu_test.txt"
  embeddings_file = "test/fixtures/embeddings_test.vec"
  connlu_sentences = load_connlu_file(connlu_file)
  system = ArcEager.ArcEagerSystem()
  settings = Settings()

  model = Model(settings, system, embeddings_file, connlu_sentences)

  @test length(model.embeddings[1, :]) == settings.embedding_size
  @test length(model.hidden_layer.weight[:, 1]) == settings.hidden_size
  @test length(model.hidden_layer.weight[1, :]) == settings.batch_size * settings.embedding_size
  @test length(model.softmax_layer.weight[:, 1]) == ArcEager.transitions_number(system)
end