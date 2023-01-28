export fetch_cache, write_cache, read_cache

using JLD2

const DEFAULT_CACHE_PATH = joinpath(@__DIR__, "..", "..", "tmp", "cache", "cache.jld2")

function fetch_cache(func::Function, cache_key; file_path=DEFAULT_CACHE_PATH)
  local prepared = nothing

  if isfile(file_path)
    jldopen(file_path, "r") do file
      if haskey(file, cache_key)
        prepared = file[cache_key]
      end
    end
  end
  if isnothing(prepared)
    prepared = func()
    jldopen(file_path, "a+") do file
      file[cache_key] = prepared
    end
  end
  
  prepared
end

function write_cache(key, data; file_path=DEFAULT_CACHE_PATH)
  jldopen(file_path, "a+") do file
    file[key] = data
  end
end

function read_cache(key; file_path=DEFAULT_CACHE_PATH)
  jldopen(file_path, "a+") do file
    file[key]
  end
end
