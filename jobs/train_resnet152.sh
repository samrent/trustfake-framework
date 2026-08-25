cd ../

python src/train.py \
    experiment.name=baselines \
    model=resnet_152

python src/test.py \
    experiment.name=baselines \
    model=resnet_152 \
    wrapper=base \
    uncertainty_score=multiclass_max_probability \
    +attack=fgsm

python src/test.py \
    experiment.name=baselines \
    model=resnet_152 \
    wrapper=mc_dropout \
    num_samples=5 \
    uncertainty_score=mc_dropout_predictive_entropy \
    +attack=fgsm \

python src/test.py \
    experiment.name=baselines \
    model=resnet_152 \
    wrapper=base \
    uncertainty_score=multiclass_max_probability \
    +attack=uncertainty_fgsm \

python src/test.py \
    experiment.name=baselines \
    model=resnet_152 \
    wrapper=mc_dropout \
    num_samples=5 \
    uncertainty_score=mc_dropout_predictive_entropy \
    +attack=uncertainty_fgsm \

# ACE is not implemented yet (src/trustfake/attacks/ace.py: __call__ is a stub),
# so these runs would evaluate against a None perturbation. Re-enable once it lands.
# python src/test.py \
#     experiment.name=baselines \
#     model=resnet_152 \
#     wrapper=base \
#     uncertainty_score=multiclass_max_probability \
#     +attack=ace

# python src/test.py \
#     experiment.name=baselines \
#     model=resnet_152 \
#     wrapper=mc_dropout \
#     num_samples=5 \
#     uncertainty_score=mc_dropout_predictive_entropy \
#     +attack=ace \
