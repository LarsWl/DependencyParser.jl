export ArcEagerSystem
export execute_transition, is_transition_valid

struct ArcEagerSystem <: ParsingSystem end;

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