@echo off
echo Starting script...

@REM docker build --tag ic-registry.epfl.ch/mr-pezeu/epfl-chatbot-compose-backend-test-local -f .\Dockerfile_local_gpu_TITANX .
docker push ic-registry.epfl.ch/mr-pezeu/epfl-chatbot-compose-backend-test-local
kubectl delete pod -l app=epfl-chatbot-compose-backend-test

echo Waiting for pod...
:waitloop
echo Checking pod status...
for /f "tokens=3" %%i in ('kubectl get pods -l app^=epfl-chatbot-compose-backend-test ^| findstr "backend-test"') do (
    echo Status found: %%i
    if "%%i"=="Running" (
        echo Pod is now running. Getting logs...
        kubectl logs -f -l app=epfl-chatbot-compose-backend-test
        goto :eof
    ) else (
        echo Waiting for pod to be running... Current status: %%i
        timeout /t 5 /nobreak >nul
        goto waitloop
    )
)