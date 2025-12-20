/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.runtime.compress.colgroup;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;
import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUtils.P;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IdentityDictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.MatrixBlockDictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.indexes.RangeIndex;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffsetIterator;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.compress.colgroup.scheme.DDCScheme;
import org.apache.sysds.runtime.compress.colgroup.scheme.ICLAScheme;
import org.apache.sysds.runtime.compress.cost.ComputationCostEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.estim.EstimationFactors;
import org.apache.sysds.runtime.compress.estim.encoding.EncodingFactory;
import org.apache.sysds.runtime.compress.estim.encoding.IEncode;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.data.SparseRow;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.jboss.netty.handler.codec.compression.CompressionException;
import shaded.parquet.it.unimi.dsi.fastutil.longs.Long2IntLinkedOpenHashMap;

/* Column Group that is based on DDC but additionally LZW compresses the
 * mapping data (_data). */
public class ColGroupDDCLZW extends ColGroupDDC {
    // private constructor. calls super(...), saves data, runs consistency checks. e.g. data not empty.
    private ColGroupDDCLZW(IColIndex colIndexes, IDictionary dict, AMapToData data, int[] cachedCounts) {
        super(colIndexes, dict, data, cachedCounts);
    }

    /**
     * Builds a packed 64-bit key for (prefixCode, nextSymbol) pairs used in the LZW dictionary.
     * Upper 32 bits: prefixCode (current pattern code w)
     * Lower 32 bits: nextSymbol (k)
     */
    private static long packKey(int prefixCode, int nextSymbol) {
        return (((long) prefixCode) << 32) | (nextSymbol & 0xffffffffL);
    }

    /**
     * LZW-compress the DDC mapping vector (AMapToData).
     * <p>
     * Input alphabet: mapping IDs in [0, unique-1] where unique = data.getUnique().
     * Output: an array of LZW codes (variable-length code stream represented as int[]).
     */
    public static int[] LZWCompressedAMapToData(final AMapToData data) {
        final int dataSize = data.size(); // number of rows / mapping entries
        final int dataUniqueVals = data.getUnique(); // number of unique values in dictionary

        // Materialize mapping indices into a flat int[] for simplicity/debuggability.
        // (For a later optimization, we can stream directly from map.getIndex(i) without this array.)
        final int[] dataIntArray = new int[dataSize];
        for (int i = 0; i < dataSize; i++) {
            dataIntArray[i] = data.getIndex(i);
        }

        // FastUtil Dictionary mapping (prefixCode, nextSymbol) pairs to LZW codes.
        // Uses a primitive Long -> int open-addressing hash map to avoid boxing and keep lookups fast.
        final Long2IntLinkedOpenHashMap dict = new Long2IntLinkedOpenHashMap();
        dict.defaultReturnValue(-1);

        // Next available code
        int nextCode = dataUniqueVals;

        // Output codes.
        final int[] out = new int[dataSize];
        int outLen = 0;

        // Current Pattern 'w' represented as its code.
        int wCode = dataIntArray[0];

        // Process remaining symbols k.
        for (int i = 1; i < dataSize; i++) {
            final int k = dataIntArray[i];
            final long wkKey = packKey(wCode, k);

            final int wkCode = dict.get(wkKey);
            if (wkCode != -1) {
                // Pattern (w+k) already exists -> extend current pattern: w = w+k.
                wCode = wkCode;
            } else {
                // Output current pattern code (w).
                out[outLen++] = wCode;

                // ADd new Pattern (w+k) to dictionary with a fresh code.
                dict.put(wkKey, nextCode++);

                // Reset pattern to the single symbol k.
                wCode = k;
            }
        }

        // Flush the last pattern,
        out[outLen++] = wCode;

        // Trim output to actual length.
        final int[] compressed = new int[outLen];
        System.arraycopy(out, 0, compressed, 0, outLen);
        return compressed;
    }

    // TODO: convert normal AMapToData into LZW-based mapping.keep special cases consistent with ColGroupDDC
    public static AColGroup create(IColIndex colIndexes, IDictionary dict, AMapToData data, int[] cachedCounts) {
        if (data.getUnique() == 1)
            // if all values in ColGroup are the same, generate ColGroup with only one tuple
            return ColGroupConst.create(colIndexes, dict);
        else if (dict == null)
            // if dictionary is empty create empty ColGroup
            return new ColGroupEmpty(colIndexes);

        // TODO: compress data to lzw variant.
        int[] lzwCodes = LZWCompressedAMapToData(data);

        // TODO:

        return new ColGroupDDCLZW(colIndexes, lzwCodes, data, cachedCounts);
    }

    // TODO: add a new value DDCLZW to CompressionType and return here to identify.
    @Override
    public CompressionType getCompType() {
        return CompressionType.DDC; // TODO: add new compression type
    }

    // TODO: add new value DDCLZW to ColGroupType and return so SystemDS can distinguish.
    @Override
    public ColGroupType getColGroupType() {
        return ColGroupType.DDC; // TODO: add new col group type
    }

    // TODO: ensure LZW mapping is written correctly.
    @Override
    public void write(DataOutput out) throws IOException {
        super.write(out);
        _data.write(out);
    }

    // TODO: needs to return ColGroupDDCLZW and reconstruct an LZW-based mapping from stream.
    @Override
    public static ColGroupDDC read(DataInput in) throws IOException {
        IColIndex cols = ColIndexFactory.read(in);
        IDictionary dict = DictionaryFactory.read(in);
        AMapToData data = MapToFactory.readIn(in);
        return new ColGroupDDC(cols, dict, data, null);
    }
}
