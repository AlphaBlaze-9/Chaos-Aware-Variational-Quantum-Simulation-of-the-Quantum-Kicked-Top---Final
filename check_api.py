from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()

for backend in service.backends():
    print(f"Backend name: {backend.name}, Status: {backend.status().operational}")