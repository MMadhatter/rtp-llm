#!/usr/bin/env bash
set -euo pipefail
# Development only: issue one experiment CA and separate server/client leaves.
out=${1:?usage: create-test-pki.sh NEW_DIRECTORY OWNER_IP}
ip=${2:?OWNER_IP required}
[[ $ip =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]
umask 077
mkdir "$out"
openssl req -x509 -newkey rsa:2048 -nodes -days 7 -subj /CN=engram-experiment-ca -keyout "$out/ca.key" -out "$out/ca.crt" >/dev/null 2>&1
for role in owner consumer producer; do
  openssl req -newkey rsa:2048 -nodes -subj "/CN=engram-$role" -keyout "$out/$role.key" -out "$out/$role.csr" >/dev/null 2>&1
  if [[ $role == owner ]]; then
    printf 'subjectAltName=IP:%s,IP:127.0.0.1\nextendedKeyUsage=serverAuth\n' "$ip" > "$out/$role.ext"
  else
    printf 'extendedKeyUsage=clientAuth\n' > "$out/$role.ext"
  fi
  openssl x509 -req -in "$out/$role.csr" -CA "$out/ca.crt" -CAkey "$out/ca.key" -CAcreateserial -days 7 -extfile "$out/$role.ext" -out "$out/$role.crt" >/dev/null 2>&1
done
printf 'Test PKI created. Deliver only ca.crt plus the relevant leaf cert/key; never distribute ca.key.\n'
