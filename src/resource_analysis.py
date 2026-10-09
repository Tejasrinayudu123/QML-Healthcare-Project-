def analyze_resources(report_list):
    for r in report_list:
        print(f"{r.name}: qubits={r.n_qubits}, depth={r.circuit_depth}, time={r.train_time_sec}")
