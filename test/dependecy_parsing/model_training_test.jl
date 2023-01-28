using DependencyParser
using DependencyParser.DependencyParsing
using DependencyParser.DependencyParsing.ArcEager

cd("test")

system = ArcEager.ArcEagerSystem()
train_file = "fixtures/connlu_test.txt"
test_file = "fixtures/connlu_test.txt"
results_file = "data"
model_file = "data/model.bson"
settings = Settings(embeddings_size=100)
connlu_sentences = load_connlu_file(train_file)
training_context = DependencyParsing.TrainingContext(
  system,
  settings,
  train_file,
  connlu_sentences,
  connlu_sentences,
  test_file,
  results_file,
  model_file,
  beam_coef = 0.05
)
model = Model(settings, system, connlu_sentences)

sort(collect(model.label_ids), by=pair->pair[end]) |>
        pairs -> map(pair -> pair[begin], pairs) |>
        labels -> set_labels!(system, labels)

@testset "Testing building dataset" begin
  dataset = DependencyParsing.build_dataset(model, connlu_sentences, training_context)

  @info size(first(first(dataset))) == 32
end