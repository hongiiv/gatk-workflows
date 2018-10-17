import "seq-format-conversion/paired-fastq-to-unmapped-bam.wdl" as PreProcessFastq
import "gatk4-data-processing/processing-for-variant-discovery-gatk4.wdl" as PreProcess
import "gatk4-somatic-snvs-indels/mutect2.wdl" as M2

workflow FullPipeline {

  ### Preprocessing fastq parameters
  Array[String] sample_name 
  Array[String] fastq_1 
  Array[String] fastq_2 
  Array[String] readgroup_name 
  Array[String] library_name 
  Array[String] platform_unit 
  Array[String] run_date 
  Array[String] platform_name 
  Array[String] sequencing_center 
  String sequencing_centers
  String ubam_list_name
  String? gatk_docker_override
  String? gatk_docker
  String? gatk_path_override
  String? gatk_path
  Int? preemptible_attempts  

  ### Preprocessing paramters
  String ref_name
  String sample_name_str
  File? flowcell_unmapped_bams_list
  String unmapped_bam_suffix
  File ref_fasta
  File ref_fasta_index
  File ref_dict
  String bwa_commandline
  Int compression_level
  File dbSNP_vcf
  File dbSNP_vcf_index
  Array[File] known_indels_sites_VCFs
  Array[File] known_indels_sites_indices
  String gotc_docker
  String python_docker
  String gotc_path
  #String gatk_path
  Int flowcell_small_disk
  Int flowcell_medium_disk
  Int agg_small_disk
  Int agg_medium_disk
  Int agg_large_disk
  Int agg_preemptible_tries

  #### M2 parameters
  File? pon
  File? pon_index
  Int scatter_count
  File? gnomad
  File? gnomad_index
  File? variants_for_contamination
  File? variants_for_contamination_index
  Boolean is_run_orientation_bias_filter = true
  Boolean is_run_oncotator = true

  File? onco_ds_tar_gz
  String? onco_ds_local_db_dir
  Array[String] artifact_modes
  String? m2_extra_args
  String? m2_extra_filtering_args
  String? sequence_source
  File? default_config_file

  String basic_bash_docker = "ubuntu:16.04"
  String oncotator_docker = "broadinstitute/oncotator:1.9.6.1"
  Int preemptible_tries
  File? gatk4_jar_override
  File wgs_coverage_interval_list
  
  call PreProcessFastq.ConvertPairedFastQsToUnmappedBamWf as PreProcessFastqs {
    input:
      sample_name = sample_name,
      fastq_1 = fastq_1,
      fastq_2 = fastq_2,
      readgroup_name = readgroup_name,
      library_name = library_name,
      platform_unit = platform_unit,
      run_date = run_date,
      platform_name = platform_name,
      sequencing_center = sequencing_center,
      ubam_list_name = ubam_list_name,
      gatk_docker_override = gatk_docker_override,
      gatk_path_override = gatk_path_override,
      preemptible_attempts = preemptible_attempts
  }
  
  call PreProcess.PreProcessingForVariantDiscovery_GATK4 as PreProcessTumor {
    input:
      sample_name = sample_name_str,
      ref_name = ref_name,
      flowcell_unmapped_bams_list = PreProcessFastqs.unmapped_bam_list,
      unmapped_bam_suffix = unmapped_bam_suffix,
      ref_fasta = ref_fasta,
      ref_fasta_index = ref_fasta_index,
      ref_dict = ref_dict,
      bwa_commandline = bwa_commandline,
      compression_level = compression_level,
      dbSNP_vcf = dbSNP_vcf,
      dbSNP_vcf_index = dbSNP_vcf_index,
      known_indels_sites_VCFs = known_indels_sites_VCFs,
      known_indels_sites_indices = known_indels_sites_indices,
      gotc_docker = gotc_docker,
      gatk_docker = gatk_docker,
      python_docker = python_docker,
      gotc_path = gotc_path,
      gatk_path = gatk_path,
      flowcell_small_disk = flowcell_small_disk,
      flowcell_medium_disk = flowcell_medium_disk,
      agg_small_disk = agg_small_disk,
      agg_medium_disk = agg_medium_disk,
      agg_large_disk = agg_large_disk,
      preemptible_tries = preemptible_tries,
      agg_preemptible_tries = agg_preemptible_tries
  }

  
    call M2.Mutect2 as M2Pair {
        input:
            intervals = wgs_coverage_interval_list,
            tumor_bam = PreProcessTumor.analysis_ready_bam,
            tumor_bai = PreProcessTumor.analysis_ready_bam_index,
            pon = pon,
            pon_index = pon_index,
            scatter_count = scatter_count,
            gnomad = gnomad,
            gnomad_index = gnomad_index,
            variants_for_contamination = variants_for_contamination,
            variants_for_contamination_index = variants_for_contamination_index,
            run_orientation_bias_filter = is_run_orientation_bias_filter,
            run_oncotator = is_run_oncotator,

            gatk_override = gatk4_jar_override,
            onco_ds_tar_gz = onco_ds_tar_gz,
            onco_ds_local_db_dir = onco_ds_local_db_dir,
            artifact_modes = artifact_modes,
            m2_extra_args = m2_extra_args,
            m2_extra_filtering_args = m2_extra_filtering_args,
            sequencing_center = sequencing_centers,
            sequence_source = sequence_source,
            default_config_file = default_config_file,

            preemptible_attempts = preemptible_attempts,
            gatk_docker = gatk_docker,
            basic_bash_docker = basic_bash_docker,
            oncotator_docker = oncotator_docker,

            ref_fasta = ref_fasta,
            ref_fai = ref_fasta_index,
            ref_dict = ref_dict,

            emergency_extra_disk = 20
    }

}
