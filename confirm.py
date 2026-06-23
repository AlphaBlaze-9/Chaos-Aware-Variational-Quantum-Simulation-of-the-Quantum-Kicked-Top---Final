from qiskit_ibm_runtime import QiskitRuntimeService

# Make sure to replace the token string below with your NEW regenerated token
QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform", 
    token="5-zIqS8jijbMV7vEHCAex3muw8SblvmRHtk07tLqJsjT", 
    overwrite=True
)

# Optional: Verify it saved correctly
service = QiskitRuntimeService()
print("Successfully loaded account on channel:", service.channel)