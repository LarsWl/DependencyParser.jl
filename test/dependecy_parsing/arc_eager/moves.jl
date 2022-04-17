using DependencyParser.DependencyParsing.ArcEager

@testset "ArcEager Moves" begin
  system = ArcEagerSystem()
  shift = Shift()
  left_arc = LeftArc()
  right_arc = RightArc()
  reduce = Reduce()


  @testset "SHIFT move" begin
    @testset "Validation" begin
      config = build_configuration()
      @test is_valid(config, shift)
      config.unshiftable[config.buffer[begin]] = true
      @test is_valid(config, shift) == false
      config.unshiftable[config.buffer[begin]] = false
      config.buffer = Vector{Integer}([1])
      @test is_valid(config, shift) == false
      config.stack = Stack{Integer}()
      @test is_valid(config, shift)
    end
    

    @testset "Transition" begin
      config =build_configuration()
      buffer_begin = config.buffer[begin]
      buffer_second = config.buffer[begin + 1]
      transition(config, "some", shift)

      @test config.buffer[begin] == buffer_second
      @test first(config.stack) == buffer_begin
    end

    @testset "Cost" begin
      config = build_configuration()
      gold_state = build_updated_gold_state(config)
      correct_cost = 0

      @test cost(gold_state, shift, "") == correct_cost
    end
  end

  @testset "REDUCE move" begin
    @testset "Validation" begin
      config = build_configuration()
      @test is_valid(config, reduce)
      config.buffer = Vector{Integer}()
      @test is_valid(config, reduce) == false
      config = build_configuration()
      config.stack = Stack{Integer}()
      @test is_valid(config, reduce) == false
    end

    @testset "Transition" begin
      #test its pop stack
      config = build_configuration()
      stack_length = length(config.stack)
      transition(config, "", reduce)
      @test length(config.stack) == stack_length - 1
      #test it unshift
      config = build_configuration()
      config.tree.nodes[first(config.stack)].head_id = DependencyParser.DependencyParsing.EMPTY_NODE
      stack_length = length(config.stack)
      buffer_length = length(config.buffer)
      transition(config, "", reduce)
      @test length(config.stack) == stack_length - 1
      @test length(config.buffer) == buffer_length + 1
      @test config.unshiftable[config.buffer[begin]]
    end

    @testset "Cost" begin
      config = build_configuration()
      gold_state = build_updated_gold_state(config)
      correct_cost = 0
      @test cost(gold_state, reduce, "") == correct_cost
      pop!(config.stack)
      gold_state = build_updated_gold_state(config)
      correct_cost = 2
      @test cost(gold_state, reduce, "") == correct_cost
    end
  end

  @testset "RIGHT ARC move" begin
    @testset "Validation" begin
      config = build_configuration()
      @test is_valid(config, right_arc)
      config.buffer = Vector{Integer}()
      @test is_valid(config, right_arc) == false
      config = build_configuration()
      config.stack = Stack{Integer}()
      @test is_valid(config, right_arc) == false
    end

    @testset "Transition" begin
      config = build_configuration()
      label = "LABEL"
      s0 = first(config.stack)
      b0 = config.buffer[begin]
      buffer_length = length(config.buffer)
      stack_length = length(config.stack)
      transition(config, label, right_arc)

      @test length(config.buffer) == buffer_length - 1
      @test length(config.stack) == stack_length + 1
      @test config.tree.nodes[b0].head_id == s0
      @test config.tree.nodes[b0].label == label
    end

    @testset "Cost" begin
      config = build_configuration()
      gold_state = build_updated_gold_state(config)
      correct_cost = -1
      @test cost(gold_state, right_arc, "SOME") == correct_cost
    end
  end

  @testset "LEFT ARC move" begin
    @testset "Validation" begin
      config = build_configuration()
      @test is_valid(config, left_arc)
      config.buffer = Vector{Integer}()
      @test is_valid(config, left_arc) == false
      config = build_configuration()
      config.stack = Stack{Integer}()
      @test is_valid(config, left_arc) == false
    end

    @testset "Transition" begin
      config = build_configuration()
      label = "LABEL"
      s0 = first(config.stack)
      b0 = config.buffer[begin]
      buffer_length = length(config.buffer)
      stack_length = length(config.stack)
      transition(config, label, left_arc)

      @test length(config.buffer) == buffer_length
      @test length(config.stack) == stack_length - 1
      @test config.tree.nodes[s0].head_id == b0
      @test config.tree.nodes[s0].label == label
    end

    @testset "Cost" begin
      config = build_configuration()
      gold_state = build_updated_gold_state(config)
      correct_cost = 1
      @test cost(gold_state, left_arc, "SOME") == correct_cost
    end
  end
end