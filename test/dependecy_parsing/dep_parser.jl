using DependencyParser.DependencyParsing

@testset "DepParser" begin
  parser = build_dep_parser()
  config = build_configuration()

  @testset "Transition prediction" begin
    correct_batch = build_correct_batch(parser.model, config)
    model_prediction = predict(parser.model, correct_batch)
    correct_transition = findmax(model_prediction) |> max_score_wiht_index -> parser.parsing_system.transitions[max_score_wiht_index[end]]

    @test predict_transition(parser, config) == correct_transition
  end
end