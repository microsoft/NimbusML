// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;
using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace Microsoft.ML.DotNetBridge
{
    internal ref struct ValueListBuilder<T>
    {
        private Span<T> _span;
        private T[] _arrayFromPool;
        private int _pos;

        public ValueListBuilder(Span<T> initialSpan)
        {
            _span = initialSpan;
            _arrayFromPool = null;
            _pos = 0;
        }

        public ValueListBuilder(int initialSize = 1024)
        {
            _arrayFromPool = ArrayPool<T>.Shared.Rent(initialSize);
            _span = _arrayFromPool;
            _pos = 0;
        }

        public int Length
        {
            get => _pos;
            set
            {
                Debug.Assert(value >= 0);
                Debug.Assert(value <= _span.Length);
                _pos = value;
            }
        }

        public int Capacity
        {
            get => _span.Length;
        }

        public ref T this[int index]
        {
            get
            {
                Debug.Assert(index < _pos);
                return ref _span[index];
            }
        }

        public T[] Buffer
        {
            get => _arrayFromPool;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Append(T item)
        {
            int pos = _pos;
            if (pos >= _span.Length)
                Grow();

            _span[pos] = item;
            _pos = pos + 1;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void AppendRange(T[] items)
        {
            int pos = _pos;
            while(pos + items.Length >= _span.Length)
                Grow();

            foreach (T item in items)
            {
                _span[pos] = item;
                _pos = pos + 1;
                pos++;
            }
        }

        public ReadOnlySpan<T> AsSpan()
        {
            return _span.Slice(0, _pos);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Dispose()
        {
            if (_arrayFromPool != null)
            {
                ArrayPool<T>.Shared.Return(_arrayFromPool);
                _arrayFromPool = null;
            }
        }

        public void Grow()
        {
            T[] array = ArrayPool<T>.Shared.Rent(_span.Length * 2);

            bool success = _span.TryCopyTo(array);
            Debug.Assert(success);

            T[] toReturn = _arrayFromPool;
            _span = _arrayFromPool = array;
            if (toReturn != null)
            {
                ArrayPool<T>.Shared.Return(toReturn);
            }
        }
    }
}