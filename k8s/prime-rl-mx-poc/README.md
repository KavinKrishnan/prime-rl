# PRIME-RL ModelExpress Weight Broadcast POC — K8s Deployment

Deploys the PRIME-RL MX weight broadcast POC on GB200 (arm64) in the `kavin` namespace.

## Prerequisites

- Existing `modelexpress-server` and `redis` services running in `kavin` namespace
- `shared-model-cache` PVC mounted with model weights
- ARM64 container image built and pushed: `nvcr.io/nvidian/dynamo-dev/prime-rl-mx:latest`

## Deploy

```bash
# Create the config
kubectl apply -f config.yaml

# Deploy inference first (needs to be ready before orchestrator starts)
kubectl apply -f inference.yaml

# Deploy trainer
kubectl apply -f trainer.yaml

# Deploy orchestrator (starts the RL loop)
kubectl apply -f orchestrator.yaml
```

## Verify MX Weight Transfer

```bash
# Check MX server logs for PublishMetadata calls from trainer
kubectl logs modelexpress-server-58f46cc95b-qctdj -n kavin | grep -i publish

# Check trainer logs for MX broadcast
kubectl logs prime-rl-trainer-0 -n kavin | grep -i "MX\|modelexpress\|broadcast"

# Check inference logs for RDMA receive
kubectl logs prime-rl-inference-0 -n kavin | grep -i "MX\|RDMA\|receive\|modelexpress"
```

## Teardown

```bash
kubectl delete -f orchestrator.yaml -f trainer.yaml -f inference.yaml -f config.yaml
```
