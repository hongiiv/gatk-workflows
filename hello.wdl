workflow myWorkflow {
    call myTask
}

task myTask {
    command {
        echo "hello world"
        sleep 10
        uname -a
    }
    output {
        String out = read_string(stdout())
    }
    runtime {
    docker: "ubuntu:latest"
    cpu: "4"
    memory: "10G"
    }
}
