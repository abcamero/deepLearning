import { useState } from "react";
import ItineraryCard from "../components/ItineraryCard";

type ItineraryItem = {
  day: number;
  title: string;
  description: string;
  location: string;
  category: string;
};

type ItineraryResponse = {
  destination: string;
  total_days: number;
  chaos_level: number;
  itinerary: ItineraryItem[];
  notes: string;
};

export default function Home() {
  const [city, setCity] = useState("Tokyo");
  const [days, setDays] = useState(5);
  const [chaos, setChaos] = useState(7);
  const [interests, setInterests] = useState("weird attractions, nightlife, local food");
  const [response, setResponse] = useState<ItineraryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setLoading(true);
    setError("");

    const body = {
      user_id: "demo_user",
      preference: {
        city,
        days,
        chaos_level: chaos,
        interests: interests.split(",").map((item) => item.trim()),
      },
    };

    try {
      const res = await fetch("http://localhost:8000/api/itineraries", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        throw new Error("Failed to fetch itinerary");
      }
      const json = await res.json();
      setResponse(json);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-900 text-slate-100">
      <div className="mx-auto max-w-6xl px-4 py-12">
        <h1 className="text-4xl font-bold">Crazy Travel Itinerary Generator</h1>
        <p className="mt-3 text-slate-300">Build a chaotic, surprise-driven itinerary using travel APIs and AI logic.</p>

        <form onSubmit={handleSubmit} className="mt-8 grid gap-6 rounded-3xl border border-slate-700 bg-slate-950/70 p-8 shadow-lg shadow-slate-950/40">
          <div className="grid gap-4 sm:grid-cols-3">
            <label className="block">
              <span className="text-slate-400">Destination</span>
              <input value={city} onChange={(e) => setCity(e.target.value)} className="mt-2 w-full rounded-xl border border-slate-700 bg-slate-900 px-4 py-3 text-white outline-none focus:border-cyan-400" />
            </label>
            <label className="block">
              <span className="text-slate-400">Days</span>
              <input type="number" value={days} min={1} onChange={(e) => setDays(Number(e.target.value))} className="mt-2 w-full rounded-xl border border-slate-700 bg-slate-900 px-4 py-3 text-white outline-none focus:border-cyan-400" />
            </label>
            <label className="block">
              <span className="text-slate-400">Chaos Level</span>
              <input type="number" value={chaos} min={1} max={10} onChange={(e) => setChaos(Number(e.target.value))} className="mt-2 w-full rounded-xl border border-slate-700 bg-slate-900 px-4 py-3 text-white outline-none focus:border-cyan-400" />
            </label>
          </div>

          <label className="block">
            <span className="text-slate-400">Interests</span>
            <input value={interests} onChange={(e) => setInterests(e.target.value)} className="mt-2 w-full rounded-xl border border-slate-700 bg-slate-900 px-4 py-3 text-white outline-none focus:border-cyan-400" />
            <p className="mt-2 text-sm text-slate-500">Separate interests with commas, e.g. weird attractions, hidden markets, photo ops.</p>
          </label>

          <button type="submit" className="inline-flex items-center justify-center rounded-2xl bg-cyan-500 px-6 py-3 text-base font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-60" disabled={loading}>
            {loading ? "Generating…" : "Generate Itinerary"}
          </button>
          {error && <p className="text-sm text-red-400">{error}</p>}
        </form>

        {response && (
          <section className="mt-10 space-y-6">
            <div className="rounded-3xl border border-slate-700 bg-slate-950/70 p-6">
              <h2 className="text-2xl font-semibold">Your Itinerary for {response.destination}</h2>
              <p className="mt-2 text-slate-400">Notes: {response.notes}</p>
            </div>
            <div className="grid gap-6">
              {response.itinerary.map((item) => (
                <ItineraryCard key={item.day} item={item} />
              ))}
            </div>
          </section>
        )}
      </div>
    </main>
  );
}
