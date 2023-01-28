mutable struct TrainingContext
    system::ParsingSystem
    settings::Settings
    train_connlu_file::String
    connlu_sentences::Vector{ConnluSentence}
    test_connlu_sentences::Vector{ConnluSentence}
    test_connlu_file::String
    training_results_file::String
    model_file::String
    optimizer
    best_uas
    best_las
    best_loss
    test_dataset
    beam_coef


    TrainingContext(
        system::ParsingSystem,
        settings::Settings,
        train_connlu_file::String,
        connlu_sentences::Vector{ConnluSentence},
        test_connlu_sentences::Vector{ConnluSentence},
        test_connlu_file::String,
        training_results_file::String,
        model_file::String;
        beam_coef = 0.05
    ) = 
    new(
        system, 
        settings,
        train_connlu_file,
        connlu_sentences,
        test_connlu_sentences,
        test_connlu_file,
        training_results_file,
        model_file,
        undef,
        0,
        0,
        Inf,
        undef,
        beam_coef
    )
end