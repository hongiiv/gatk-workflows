## The new standard in cloud bioinformatics (WDL+Cromwell)

### Installing Google Cloud SDK

### Installing MySQL database

### Authorizing Google Cloud

`# gcloud init`

`# gcloud auth login <login-id>`

`# gcloud auth application-default login`

`# gcloud config set project <google-cloud-project-id>`

`# git clone https://github.com/hongiiv/gatk-workflows.git`

`# cd gatk-workflows`

### Edit application options (edit google.conf, generic.google-papi.options.json)

### Run

`# java -Dconfig.file=google.conf -jar cromwell-35.jar run hello.wdl -o generic.google-papi.options.json`

	# java -Dconfig.file=google.conf -jar cromwell-35.jar \
	submit FullPipeline.wdl \
	-I FullPipeline.input.json \
	-o generic.google-papi-options.json \
	-h http://localhost:8080 \
	-p workflowDependencies.zip
