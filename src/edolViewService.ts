import * as vscode from 'vscode';
import * as zlib from 'zlib';
import * as net from 'net';
import * as path from 'path';
import * as fs from 'fs/promises';
import { pythonCodeBuilder } from './pythonCodeBuilder';
import CvVariable from './CvVariable';
import PromiseSocket from './PromiseSocket';

interface Extra {
    nbytes: number;
    dtype: number;
    shape: [number, number, number];
    compression: string; // "png" | "zlib" | "cv"
}


export class EdolViewImageHandler{

    constructor() {
    }

    private serializeExtra(extra: Extra): Buffer {
        // Serialize Extra struct to binary format
        // u64 nbytes (8 bytes) + u32 dtype (4 bytes) + [u32; 3] shape (12 bytes) + String compression (length + data)
        
        const compressionBuffer = Buffer.from(extra.compression, 'utf-8');
        const compressionLength = compressionBuffer.length;
        
        // Total size: 8 + 12 + 4 + compressionLength
        const buffer = Buffer.allocUnsafe(8 + 12 + 4 + compressionLength);
        let offset = 0;
        
        // Write nbytes as u64 (8 bytes)
        buffer.writeBigUInt64BE(BigInt(extra.nbytes), offset);
        offset += 8;
        
        // Write dtype as u32 (4 bytes)
        buffer.writeUInt32BE(extra.dtype, offset);
        offset += 4;
        
        // Write shape as [u32; 3] (12 bytes total)
        buffer.writeUInt32BE(extra.shape[0], offset);
        offset += 4;
        buffer.writeUInt32BE(extra.shape[1], offset);
        offset += 4;
        buffer.writeUInt32BE(extra.shape[2], offset);
        offset += 4;
        
        // Write compression string data
        compressionBuffer.copy(buffer, offset);
        
        return buffer;
    }

    async sendData(name: string, extra: Extra, data: Buffer) {

        const host: string = vscode.workspace.getConfiguration().get("edolview.host") ?? "127.0.0.1";
        const port: number = vscode.workspace.getConfiguration().get("edolview.port") ?? 21734;

        const netSocket = new net.Socket();
        
        const socket = new PromiseSocket(netSocket);
        await socket.connectPromise(port, host);

        const extraBuffer = this.serializeExtra(extra);

        const lengthBuffer = await Buffer.allocUnsafe(24);
        let offset = 0;
        offset = lengthBuffer.writeBigUInt64BE(BigInt(name.length), offset);
        offset = lengthBuffer.writeBigUInt64BE(BigInt(extraBuffer.length), offset);
        offset = lengthBuffer.writeBigUInt64BE(BigInt(data.length), offset);

        await socket.writeBuffer(lengthBuffer);
        
        await socket.writeStr(name);
        await socket.writeBuffer(extraBuffer);
        await socket.writeBuffer(data);

        await socket.end();

    }

    async addImageFile(filePath: string) {

        const data = await fs.readFile(filePath);

        const name = path.basename(filePath);
        const extra: Extra = {
            nbytes: 0,
            dtype: 0,
            shape: [0, 0, 0],
            compression: 'cv'
        };

        await this.sendData(name, extra, data);
    }

    async addImagePython(variable: Variable) {
        const session = vscode.debug.activeDebugSession;

        if(session) {            
            // const variables: Array<Variable> = (await session.customRequest('variables', {variablesReference: varRef})).variables;

            // const response = await session.customRequest('variables', {variablesReference: varRef});
            
            // let spVariable = response.variables.find((v: Variable) => v.name === 'special variables');
            // const spResponse = await session.customRequest('variables', {variablesReference: spVariable.variablesReference});
            
            // let internalVariable = spResponse.variables.find((v: Variable) => v.name === '__internals__');
            // const internalResponse = await session.customRequest('variables', {variablesReference: internalVariable.variablesReference});

            // let dataVariable = internalResponse.variables.find((v: Variable) => v.name === '\'data\'');
            // let ctypesVariable = internalResponse.variables.find((v: Variable) => v.name === '\'ctypes\'');
            
            // const dataResponse = await session.customRequest('variables', {variablesReference: dataVariable.variablesReference});
            // const ctypesResponse = await session.customRequest('variables', {variablesReference: ctypesVariable.variablesReference});

            // const memoryAddr = parseInt(ctypesResponse.variables.find((v: Variable) => v.name === 'data').value);
            // const nBytes = parseInt(dataResponse.variables.find((v: Variable) => v.name === 'nbytes').value);

            const threadRes = await session.customRequest('threads', {});
            const threads = threadRes.threads;

            const stackTraceRes = await session.customRequest('stackTrace', { threadId: threads[0].id });
            const stacks = stackTraceRes.stackFrames;

            const callStack = stacks[0].id;

            const host: string = vscode.workspace.getConfiguration().get("edolview.host") ?? "127.0.0.1";
            const port: number = vscode.workspace.getConfiguration().get("edolview.port") ?? 21734;
            const floatToHalf: boolean = vscode.workspace.getConfiguration().get("edolview.float_to_half") ?? false;
            const doCompression: boolean = vscode.workspace.getConfiguration().get("edolview.do_compression") ?? false;
            const downscale: number = vscode.workspace.getConfiguration().get("edolview.downscale") ?? 1;
            
            const pythonCode = pythonCodeBuilder(variable.evaluateName, host, port, floatToHalf, doCompression, downscale);

            await session.customRequest("evaluate", { expression: pythonCode, frameId: callStack, context: 'repl' });
        }
    }

    async addImageCpp(varName: string, varRef: number) {
        const session = vscode.debug.activeDebugSession;

        if(session) {
            const variables: Array<Variable> = (await session.customRequest('variables', {variablesReference: varRef})).variables;

            const cvVariable = CvVariable.parseCvVariable(variables);
            if (cvVariable !== null) {
                const buf = await cvVariable.readData(session);
            
                const compressedBuf = zlib.deflateSync(buf);
                
                const extra: Extra = {
                    nbytes: cvVariable.nbytes,
                    dtype: cvVariable.dtype,
                    shape: [cvVariable.rows, cvVariable.cols, cvVariable.channels],
                    compression: 'zlib'
                };
                
                await this.sendData(varName, extra, compressedBuf);
            }
        }
    }
}