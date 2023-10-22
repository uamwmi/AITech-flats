# Frontend AITech/flats

## Uruchomienie

Do integracji z backendem potrzebujemy jego adres i token. Najlepiej przechowywać je w pliku `.env`:

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

Strona powinna działać pod `http://localhost`.
[README.md](README.md)

## Budowanie i deployment

### Zbudowanie kontenera

```shell
$ docker build -f .\Dockerfile.prod --build-arg API_URL="{{adres_backendu}}" --build-arg API_TOKEN="{{token}}" -t "aitechflats.azurecr.io/frontend" .
```

Podmieniamy:

- Adres hosta backendu (`{{adres_backendu}}`), np. *https://aitechflats-func-inference-dev.azurewebsites.net*
- Token dostępowy do backendu (`{{token}}`) .

### Deployment

Po zmergowania do maina aplikacja automatycznie deployuje się do Azure Static Web Apps.

Strona jest podstępna pod https://yellow-pond-04f25f30f.2.azurestaticapps.net