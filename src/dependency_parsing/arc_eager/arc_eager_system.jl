export ArcEagerSystem
export execute_transition, is_transition_valid, transitions_number, set_labels!

mutable struct ArcEagerSystem <: ParsingSystem 
  labels::Vector{String}
  transitions::Vector{Transition}

  ArcEagerSystem() = new()
  function ArcEagerSystem(labels::Vector{String})
    system = new()
    set_labels!(system, labels)

    system
  end
end

function set_labels!(system::ArcEagerSystem, labels::Vector{String})
  transitions = Vector{Transition}()
    
  push!(transitions, Transition(Shift(Moves.SHIFT), EMPTY_LABEL))
  push!(transitions, Transition(Reduce(Moves.REDUCE), EMPTY_LABEL))

  foreach(labels) do label
    for move in [LeftArc(Moves.LEFT), RightArc(Moves.RIGHT)]
      push!(transitions, Transition(move, label))
    end
  end

  system.labels = labels
  system.transitions = transitions
end

function transitions_number(system::ArcEagerSystem)
  length(system.transitions)
end

function execute_transition(config::Configuration, trans::Transition, system::ArcEagerSystem)
  if !is_transition_valid(config, trans, system)
    return false
  end

  transition(config, trans.label, trans.move)

  true
end

function is_transition_valid(config::Configuration, transition::Transition, system::ArcEagerSystem)
  is_valid(config, transition.move)
end