type ItineraryItem = {
  day: number;
  title: string;
  description: string;
  location: string;
  category: string;
};

export default function ItineraryCard({ item }: { item: ItineraryItem }) {
  return (
    <article className="rounded-3xl border border-slate-700 bg-slate-950/80 p-6 shadow-lg shadow-slate-950/20">
      <div className="mb-4 flex items-center justify-between gap-3 text-slate-300">
        <span className="rounded-full bg-cyan-500/10 px-3 py-1 text-sm font-semibold text-cyan-300">Day {item.day}</span>
        <span className="text-sm uppercase tracking-[0.2em] text-slate-500">{item.category}</span>
      </div>
      <h3 className="text-xl font-semibold text-white">{item.title}</h3>
      <p className="mt-2 text-slate-400">{item.description}</p>
      <p className="mt-4 text-sm text-slate-500">Location: {item.location}</p>
    </article>
  );
}
