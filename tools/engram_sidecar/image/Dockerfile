ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY --chown=0:0 runtime/ /opt/engram-runtime/
ENTRYPOINT ["/opt/engram-runtime/repo/tools/engram_sidecar/image/entrypoint.sh"]
CMD ["help"]
