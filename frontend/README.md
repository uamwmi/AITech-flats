# Frontend AITech/flats

## Uruchomienie

To integrate with the backend, we need its address and token. It is best to store these in a `.env` file:

```shell
VITE_API_URL="https://aitechflats-func-inference-dev.azurewebsites.net"
VITE_API_TOKEN="unsafechangeme"
```

Uruchomienie:

```shell
$ npm i
$ npm run dev
```

### Docker

```shell
$ docker-compose up
```

The site should work under `http://localhost`.
[README.md](README.md)

## Budowanie i deployment

### Zbudowanie kontenera

```shell
$ docker build -f .\Dockerfile.prod --build-arg API_URL="{{adres_backendu}}" --build-arg API_TOKEN="{{token}}" -t "aitechflats.azurecr.io/frontend" .
```

Replace:

- The host address of the backend (`{{address_backend}}), e.g.. *<https://aitechflats-func-inference-dev.azurewebsites.net>*
- The access token for the backend (`{{token}}`) .

### Deployment

When merged into maina, the app automatically deploys to Azure Static Web Apps.
