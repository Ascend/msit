ENABLE=${1-1}
echo "ENABLE=$ENABLE"

if [ "$ENABLE" = "1" ]; then
    export CPU_AFFINITY_CONF=2
else
    unset CPU_AFFINITY_CONF
fi
