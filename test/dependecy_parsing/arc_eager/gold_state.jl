@testset "GoldState" begin
  tree = build_gold_tree()
  config = build_configuration()
  state = GoldState(tree, config)

  head_in_stack = DependencyParser.DependencyParsing.ArcEager.HEAD_IN_STACK
  head_in_buffer = DependencyParser.DependencyParsing.ArcEager.HEAD_IN_BUFFER

  correct_root_dependents_in_stack = 1
  correct_root_dependents_in_buffer = 0
  correct_dependents_in_stack = [0, 1, 0, 0, 0, 0]
  correct_dependents_in_buffer = [0 ,2, 0, 0, 1, 0]
  correct_heads_states = [
    head_in_stack,
    head_in_stack,
    head_in_stack,
    head_in_buffer,
    head_in_stack,
    head_in_stack
  ]

  # Expect it correct calculate dependents arcs count
  @test state.root_dependents_in_buffer_count == correct_root_dependents_in_buffer
  @test state.root_dependents_in_stack_count == correct_root_dependents_in_stack
  @test state.dependents_in_buffer_count == correct_dependents_in_buffer
  @test state.dependents_in_stack_count == correct_dependents_in_stack
  @test state.heads_states == correct_heads_states
end