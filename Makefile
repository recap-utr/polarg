.PHONY: convert
convert: convert-pheme convert-araucaria convert-microtexts convert-persuasive-essays convert-kialo

.PHONY: convert-pheme
convert-pheme:
	poetry run python -m argument_nli.convert pheme data/pheme/test_training_splits "pheme_train_*.xml" "pheme_test_*.xml" data/pheme

.PHONY: convert-araucaria
convert-araucaria:
	poetry run python -m argument_nli.convert argument-graph data/araucaria "train/*.json" "test/*.json" data/araucaria

.PHONY: convert-kialo
convert-kialo:
	poetry run python -m argument_nli.convert argument-graph data/kialo "train/*.txt" "test/*.txt" data/kialo

.PHONY: convert-microtexts
convert-microtexts:
	poetry run python -m argument_nli.convert argument-graph data/microtexts "train/*.txt" "test/*.txt" data/microtexts

.PHONY: convert-persuasive-essays
convert-persuasive-essays:
	poetry run python -m argument_nli.convert argument-graph data/persuasive-essays "train/*.txt" "test/*.txt" data/persuasive-essays

.PHONY: sync
sync:
	rsync -e ssh -ar --delete --exclude ".git" --filter=":- .gitignore" ./ gpu:~/developer/argument-nli
