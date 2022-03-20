module Units
  export AbstractPipelineUnit,
    AbstractSentenceSplitter,
    AbstractTokenizer,
    AbstractPOSTagger,
    AbstractDepParser

  abstract type AbstractPipelineUnit end;

  abstract type AbstractSentenceSplitter <: AbstractPipelineUnit end
  abstract type AbstractTokenizer <: AbstractPipelineUnit end;
  abstract type AbstractPOSTagger <: AbstractPipelineUnit end;
  abstract type AbstractDepParser <: AbstractPipelineUnit end;
end
