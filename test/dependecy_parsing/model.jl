using DependencyParser
using DependencyParser.DependencyParsing
using DependencyParser.DependencyParsing.ArcEager

cd("test")

system = ArcEager.ArcEagerSystem()
connlu_sentences = load_connlu_file("fixtures/connlu_test.txt")
settings = Settings(embeddings_size=100)


model = Model(settings, system, connlu_sentences)

@testset "Model" begin
  model = DependencyParsing.Model()
end
