module DependencyParsing
  using ..DependencyParser.Core

  include("parsing_system.jl")
  include("transition.jl")
  include("tree_node.jl")
  include("dependency_tree.jl")
  include("configuration.jl")
  include("gold_state.jl")
  include("arc_eager/arc_eager.jl")
  include("dep_parser.jl")
end