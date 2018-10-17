## The new standard in cloud bioinformatics (WDL+Cromwell)

### Prerequisites

* A Unix-based operating system (Mac/Linux)
* A Java 8 runtime environment (you can download java [here](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html))
* Google Cloud SDK (you can download GCS [here](https://cloud.google.com/sdk/install))
* A MySQL database (create database cromwell)
* On your Google project, open up the API Manager and enable the following APIs
  * Google Compute Engine
  * Google Cloud Strorage
  * Google Genomics API
  
### Download Cromwell jar

This code tested on Cromwell v35.

`# wget https://github.com/broadinstitute/cromwell/releases/download/36/cromwell-36.jar`

### Authorizing Google Cloud

`# gcloud init`

`# gcloud auth login <login-id>`

`# gcloud auth application-default login`

`# gcloud config set project <google-cloud-project-id>`


### Download gatk-workflows for somatic
`# git clone https://github.com/hongiiv/gatk-workflows.git`

copy cromwell.jar

`cp cromwell-36.jar gatk-workflows/`


### Edit application options (edit google.conf)
* MySQL database id/pw (user = your mysql id, password = your mysql p/w)
* Google project (project = your google project name)
* Base bucket (root = your google storage bucket name)

### Run Cromwell server

`# java -Dconfig.file=google.conf -jar cromwell-35.jar run hello.wdl -o generic.google-papi.options.json`

### Run workflow

	# java -Dconfig.file=google.conf -jar cromwell-35.jar \
	submit FullPipeline.wdl \
	-I FullPipeline.input.json \
	-o generic.google-papi-options.json \
	-h http://localhost:8080 \
	-p workflowDependencies.zip
