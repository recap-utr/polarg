.PHONY: convert
convert: convert-pheme, convert-araucaria, convert-kialo

.PHONY: convert-pheme
convert-pheme:
	poetry run python -m argument_nli.convert pheme data/pheme/test_training_splits "pheme_train_*.xml" "pheme_test_*.xml" data/pheme.json

.PHONY: convert-araucaria
convert-araucaria:
	poetry run python -m argument_nli.convert argument-graph data/araucaria "train/*.json" "test/*.json" data/araucaria.json

.PHONY: convert-kialo
convert-kialo:
	poetry run python -m argument_nli.convert argument-graph data/kialo "train/*.json" "test/*.json" data/kialo.json

.PHONY: sync
sync:
	rsync -e ssh -ar --delete --exclude ".git" --filter=":- .gitignore" ./ gpu:~/developer/argument-nli
