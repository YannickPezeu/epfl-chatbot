export default function Debug() {
  return (
    <pre>
      {JSON.stringify({
        NEXT_PUBLIC_BACKEND_URL: process.env.NEXT_PUBLIC_BACKEND_URL,
        env: Object.fromEntries(
          Object.entries(process.env)
        )
      }, null, 2)}
    </pre>
  );
}