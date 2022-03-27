using DependencyParser
using Test
using DataStructures

# FACTORIES

include("factories/dependency_tree.jl")
include("factories/sentence.jl")
include("factories/configuration.jl")
include("factories/gold_state.jl")

@testset "DependencyParser.jl" begin
    include("pipeline.jl")
    include("dependecy_parsing/arc_eager/moves.jl")
    include("dependecy_parsing/gold_state.jl")
end