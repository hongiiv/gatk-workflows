## The new standard in cloud bioinformatics (WDL+Cromwell)

#### Prerequisites

* A Unix-based operating system (Mac/Linux)
* A Java 8 runtime environment (you can download java [here](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html))
* Google Cloud SDK (you can download GCS [here](https://cloud.google.com/sdk/install))
* A MySQL database (create database cromwell)
* On your Google project, open up the API Manager and enable the following APIs
  * Google Compute Engine
  * Google Cloud Strorage
  * Google Genomics API
  
#### Download Cromwell jar

This code tested on Cromwell v35.

`# wget https://github.com/broadinstitute/cromwell/releases/download/36/cromwell-36.jar`

#### Authorizing Google Cloud

`# gcloud init`

`# gcloud auth login <login-id>`

`# gcloud auth application-default login`

`# gcloud config set project <google-cloud-project-id>`


#### Download gatk-workflows for somatic

`# git clone https://github.com/hongiiv/gatk-workflows.git`

copy cromwell.jar

`# cp cromwell-36.jar gatk-workflows/`

`# cd gatk-workfolws`

This workflow calls another workflow, that second workflow is referred to as a sub-workflow. This workflow based on [official GATK workflows](https://github.com/gatk-workflows/)

* [seq-format-conversion](https://github.com/gatk-workflows/seq-format-conversion): Workflows for converting between sequence data formats
* [gatk4-data-processing](https://github.com/gatk-workflows/gatk4-data-processing): Workflows for processing high-throughput sequencing data for variant discovery with GATK4 and related tools
* [gatk4-somatic-snvs-indes](https://github.com/gatk-workflows/gatk4-somatic-snvs-indels): Workflows for somatic shrot variant discovery with GATK4

`# zip -r workflowDependencies.zip ./gatk4-data-processing/ ./gatk4-somatic-snvs-indels/ ./seq-format-conversion/`


#### Edit application options (edit google.conf)
* MySQL database id/pw (user = your mysql id, password = your mysql p/w)
* Google project (project = your google project name)
* Base bucket (root = your google storage bucket name)

#### Run Cromwell server

`# java -Dconfig.file=google.conf -jar cromwell-35.jar run hello.wdl -o generic.google-papi.options.json`

#### Run workflow via Cromwell server

	# java -Dconfig.file=google.conf -jar cromwell-35.jar \
	submit FullPipeline.wdl \
	-I FullPipeline.input.json \
	-o generic.google-papi-options.json \
	-h http://localhost:8080 \
	-p workflowDependencies.zip

#### Run workflow via command
	# java -Dconfig.file=google.conf -jar cromwell-35.jar \
	run FullPipeline.wdl \
	-I FullPipeline.input.json \
	-o generic.google.papi-options.json \ 
	
#### Run workflow via REST API
	# curl -X POST "http://localhost:8080/api/workflows/v1" \
	-H "accept: application/json" -H "Content-Type: multipart/form-data" \
	-F "workflowSource=@FullPipeline.wdl;type=" \
	-F "workflowInputs=@FullPipeline.input.json;type=application/json" \
	-F "workflowDependencies=@workflowDependencies.zip;type=application/zip"

#### Run workflow via REST API using Cromwell Server

You can access swagger web page (http://localhost:8080)

#### FAQ

Q: Workflow를 실행하는 방법이 여러개가 존재합니다. 어떠한 방법을 사용하는것이 좋을까요?

A: Server 모드 (REST API 또는 submit 명령)를 통해 실행하는 것이 좋습니다. 해당 workflow를 관리가 가능하기 때문입니다.

Q: 왜 GATK workflow에서는 FASTQ 파일 대신 uBAM을 기본으로 사용합니까?

A: 최근 발표된 "Standards Guidelines for Validating Next-Generation Sequencing Bioinformatics Pipelines"에 따르면 laboratory, run, patient identifiers가 파일의 메타데이터에 기록될 것을 권고하고 있습니다. 또한 파일 이름 자체에도 이러한 식별자가 표시되어야 한다고 권고하고 있습니ㅏㄷ. 이러한 요구사항을 충족 시키기 위해서는 uBAM이 적합합니다.

Q: WDL을 지원하는 상업적인 서비스가 존재합니까?

A: 미국의 [DNAnexus](https://www.dnanexus.com)와 캐나다의 [DNAstack](https://www.dnastack.com)이 WDL을 지원하고 있습니다.

