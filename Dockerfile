# syntax=docker/dockerfile:1

FROM golang:1.26 AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
# The SQLite driver is pure Go, so the binary is fully static.
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/gateway ./cmd/gateway \
    && mkdir /out/data

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/gateway /gateway
# Owned by the runtime user so that SQLite can create its files; a named
# volume mounted here inherits this ownership.
COPY --from=build --chown=nonroot:nonroot /out/data /data

# Listening beyond loopback makes GATEWAY_API_KEY mandatory: the service
# refuses to start without it.
ENV GATEWAY_ADDR=:8080 \
    GATEWAY_DB_PATH=/data/gateway.db
VOLUME /data
EXPOSE 8080
USER nonroot
ENTRYPOINT ["/gateway"]
