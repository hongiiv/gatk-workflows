### The new standard in cloud bioinformatics (WDL+Cromwell)

#### Installing Google Cloud SDK
[https://cloud.google.com/sdk/install](https://cloud.google.com/sdk/install)

#### Installing MySQL database

#### Download Cromwell jar

`# wget https://github.com/broadinstitute/cromwell/releases/download/36/cromwell-36.jar`

#### Authorizing Google Cloud

`# gcloud init`

`# gcloud auth login <login-id>`

`# gcloud auth application-default login`

`# gcloud config set project <google-cloud-project-id>`

`# git clone https://github.com/hongiiv/gatk-workflows.git`

`# cd gatk-workflows`

#### Download gatk-workflows for somatic
`# git clone https://github.com/hongiiv/gatk-workflows.git`

copy cromwell.jar

`cp cromwell-36.jar gatk-workflows/`


#### Edit application options (edit google.conf, generic.google-papi.options.json)
* MySQL database id/pw
* Google Cloud id/pw
* Google project

#### Run Cromwell server

`# java -Dconfig.file=google.conf -jar cromwell-35.jar run hello.wdl -o generic.google-papi.options.json`

#### Run workflow

	# java -Dconfig.file=google.conf -jar cromwell-35.jar \
	submit FullPipeline.wdl \
	-I FullPipeline.input.json \
	-o generic.google-papi-options.json \
	-h http://localhost:8080 \
	-p workflowDependencies.zip
