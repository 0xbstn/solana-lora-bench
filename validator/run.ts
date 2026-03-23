import path from 'path';

async function main() {
    const [, , filePath, blockhash] = process.argv;

    if (!filePath || !blockhash) {
        console.log(JSON.stringify({ serialized_tx: null, error: 'Usage: bun run run.ts <file> <blockhash>' }));
        process.exit(1);
    }

    const absolutePath = path.resolve(filePath);

    try {
        const mod = await import(absolutePath);

        if (typeof mod.executeSkill !== 'function') {
            console.log(JSON.stringify({ serialized_tx: null, error: 'executeSkill function not found' }));
            process.exit(1);
        }

        const result = await Promise.race([
            mod.executeSkill(blockhash),
            new Promise<string>((_, reject) =>
                setTimeout(() => reject(new Error('Skill execution timed out')), 8000)
            ),
        ]);

        console.log(JSON.stringify({ serialized_tx: result }));
    } catch (error: any) {
        let errorMessage = 'Unknown error';
        let errorDetails = '';

        if (error?.name === 'AggregateError' && Array.isArray(error.errors)) {
            errorMessage = error.message || 'Multiple errors';
            errorDetails = error.errors.map((e: any) => e?.message || String(e)).join('\n');
        } else if (error instanceof Error) {
            errorMessage = error.message;
            errorDetails = error.stack || error.toString();
        } else {
            errorMessage = String(error);
        }

        console.log(JSON.stringify({
            serialized_tx: null,
            error: errorMessage,
            details: errorDetails.slice(0, 1000),
        }));
        process.exit(1);
    }
}

main();
