module ArcEager
  export Shift, Reduce, RightArc, LeftArc, Break, ArcEagerSystem
  export cost, is_valid, transition

  using ..DependencyParsing

  struct Shift <: Move end;
  struct Reduce <: Move end;
  struct RightArc <: Move end;
  struct LeftArc <: Move end;
  struct Break <: Move end;

  struct ArcEagerSystem end;

  #=
    Cost: push_cost 
  =#
  function cost(config::Configuration, move::Shift)

  end

  #=
    Validity:
    * If stack is empty
    * At least two words in sentence
    * Word has not been shifted before
  =#
  function is_valid(config::Configuration, word_id::Integer, move::Shift)
    if stack_depth(config) == 0
      return true
    elseif buffer_lenght(config) < 2
      return false
    elseif is_unshiftable(config, word_id)
      return false
    else
      return true
    end
  end

  # Move the first word of the buffer onto the stack and mark it as "shifted"
  function transition(config::Configuration, move::Shift)
    push(config)
  end

  #=
  Cost:
    * If B[0] is the start of a sentence, cost is 0
    * Arcs between stack and buffer
    * If arc has no head, we're saving arcs between S[0] and S[1:], so decrement
        cost by those arcs.
  =#
  function cost(config::Configuration, move::Reduce)
  end

  #=
    Validity:
      * Stack not empty
      * Buffer nt empty
      * Stack depth 1 and cannot senten start l_edge(st.B(0)) ?
  =#
  function is_valid(config::Configuration, move::Reduce)
    if stack_depth(config) == 0
      return false
    elseif buffer_length(config) == 0
      return false
    else
      return true
    end
  end

  # Pop from the stack. If it has no head and the stack isn't empty, place it back on the buffer.
  function transition(config::Configuration, move::Reduce)
    if has_head(config.tree, config.stack[begin]) || stack_depth(config) == 1
      pop!(config)
    else
      unshift!(config)
    end
  end



  function cost(config::Configuration, move::RightArc)
  end

  function cost(config::Configuration, move::LeftArc)
  end

  function cost(config::Configuration, move::Break)
  end


  

  function is_valid(config::Configuration, move::RightArc)
  end

  function is_valid(config::Configuration, move::LeftArc)
  end

  function is_valid(config::Configuration, move::Break)
  end

  function transition(config::Configuration, move::RightArc)
  end

  function transition(config::Configuration, move::LeftArc)
  end

  function transition(config::Configuration, move::Break)
  end
end