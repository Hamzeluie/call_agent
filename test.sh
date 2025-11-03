PORT=8000
echo "📡 Scanning for ALL processes using port $PORT..."

# Get ALL PIDs using this port (listeners AND connected clients)
PIDS=$(sudo lsof -ti :$PORT) || true

if [ -n "$PIDS" ]; then
echo "🧨 Found the following PIDs: $PIDS"
echo "📜 Details:"
lsof -i :$PORT

# Kill them ALL, one by one
for PID in $PIDS; do
    echo "💀 Killing PID $PID..."
    sudo kill -9 $PID 2>/dev/null && echo "✅ Killed $PID" || echo "⚠️ Failed to kill $PID"
done

# Final verification
if lsof -i :$PORT > /dev/null 2>&1; then
    echo "❌❌❌ EMERGENCY: Port $PORT is STILL in use after multiple kills. Manual intervention required."
    echo "📋 Run this manually on the host:"
    echo "   sudo lsof -i :${{ env.PORT }}"
    echo "   sudo kill -9 <PID>"
    exit 1
else
    echo "✅✅✅ Port $PORT is now 100% FREE. Proceeding..."
fi
else
echo "✅ Port $PORT is free — nothing to kill."
fi