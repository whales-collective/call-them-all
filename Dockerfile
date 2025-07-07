FROM golang:1.24.4-alpine AS builder
WORKDIR /app
COPY go.mod .
COPY main.go .

RUN <<EOF
go mod tidy 
go build
EOF

FROM scratch
WORKDIR /app
COPY --from=builder /app/call-them-all .
ENTRYPOINT ["./call-them-all"]
