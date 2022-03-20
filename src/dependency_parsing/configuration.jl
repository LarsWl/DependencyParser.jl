export Configuration
export stack_depth, buffer_lenght

using DataStructures

mutable struct Configuration
  buffer::Vector{Integer}
  stack::Stack{Integer}
  sentence::Vector{Tuple{String, String}}
  shifted::Vector{Bool}
  unshifted::Vector{Bool}
  tree::DependencyTree

  function Configuration(sentence::Vector{Tuple{String, String}})
    buffer = map(enumerate(sentence)) do (index, _)
      index
    end
    shifted = zeros(Bool, length(sentence))
    unshifted = zeros(Bool, length(sentence))
    stack = Stack{Integer}()
    tree = DependencyTree(sentence)
    
    new(buffer, stack, sentence, shifted, tree)
  end
end

function stack_depth(config::Configuration)
  length(config.stack)
end

function buffer_length(config::Configuration)
  length(config.buffer)
end

function push!(config::Configuration)
  if buffer_lenght(config) == 0
    return
  end
  
  buffer_element = pop!(config.buffer)
  push!(config.stack, buffer_element)
  shifted[buffer_element] = true
end

function pop!(config::Configuration)
  if stack_depth(config) == 0
    return
  end

  pop!(config.stack)
end

function unshift!(config::Configuration)
  word_id = pop!(config.stack)
  insert!(config.buffer, 1, word_id)
  config.unshifted[word_id] = true
end

function is_unshiftable(config::Configuration, word_id::Integer)
  config.shifted[word_id]
end