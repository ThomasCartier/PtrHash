evals: cpufreq
    mkdir -p data
    cargo run -r --example evals

cpufreq:
    sudo cpupower frequency-set --governor performance -d 2.6GHz -u 2.6GHz > /dev/null
