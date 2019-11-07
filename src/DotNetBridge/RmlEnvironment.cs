//------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//------------------------------------------------------------------------------

using System;
using System.Globalization;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.DotNetBridge
{
    internal class RmlEnvironment : HostEnvironmentBase<RmlEnvironment>
    {
        private Bridge.CheckCancelled CheckCancelled;

        private sealed class Channel : ChannelBase
        {
            public Channel(RmlEnvironment master, ChannelProviderBase parent, string shortName, Action<IMessageSource, ChannelMessage> dispatch)
                : base(master, parent, shortName, dispatch)
            {
            }
        }

        private sealed class Host : HostBase
        {

            public Host(HostEnvironmentBase<RmlEnvironment> source, string shortName, string parentFullName, Random rand, bool verbose)
                : base(source, shortName, parentFullName, rand, verbose)
            {
            }

            protected override IChannel CreateCommChannel(ChannelProviderBase parent, string name)
            {
                Contracts.AssertValue(parent);
                Contracts.Assert(parent is Host);
                Contracts.AssertNonEmpty(name);
                return new Channel(Root, parent, name, GetDispatchDelegate<ChannelMessage>());
            }

            protected override IPipe<TMessage> CreatePipe<TMessage>(ChannelProviderBase parent, string name)
            {
                Contracts.AssertValue(parent);
                Contracts.Assert(parent is Host);
                Contracts.AssertNonEmpty(name);
                return new Pipe<TMessage>(parent, name, GetDispatchDelegate<TMessage>());
            }

            protected override IHost RegisterCore(HostEnvironmentBase<RmlEnvironment> source, string shortName, string parentFullName, Random rand, bool verbose)
            {
                return new Host(source, shortName, parentFullName, rand, verbose);
            }
        }

        public RmlEnvironment(Bridge.CheckCancelled checkDelegate, int? seed = null, bool verbose = false)
            : this(RandomUtils.Create(seed), verbose)
        {
            CheckCancelled = checkDelegate;
        }

        public RmlEnvironment(Random rand, bool verbose = false)
            : base(rand, verbose)
        {
            CultureInfo.CurrentUICulture = CultureInfo.InvariantCulture;
            EnsureDispatcher<ChannelMessage>();
        }

        public RmlEnvironment(RmlEnvironment source, int? seed = null, bool verbose = false)
            : this(source, RandomUtils.Create(seed), verbose)
        {
        }

        public RmlEnvironment(RmlEnvironment source, Random rand, bool verbose = false)
            : base(source, rand, verbose)
        {
            CultureInfo.CurrentUICulture = CultureInfo.InvariantCulture;
            EnsureDispatcher<ChannelMessage>();
        }

        protected override IHost RegisterCore(HostEnvironmentBase<RmlEnvironment> source, string shortName, string parentFullName, Random rand, bool verbose)
        {
            Contracts.AssertValue(rand);
            Contracts.AssertValueOrNull(parentFullName);
            Contracts.AssertNonEmpty(shortName);
            Contracts.Assert(source == this || source is Host);
            return new Host(source, shortName, parentFullName, rand, verbose);
        }


        protected override IChannel CreateCommChannel(ChannelProviderBase parent, string name)
        {
            Contracts.AssertValue(parent);
            Contracts.Assert(parent is ConsoleEnvironment);
            Contracts.AssertNonEmpty(name);
            return new Channel(this, parent, name, GetDispatchDelegate<ChannelMessage>());
        }

        protected override IPipe<TMessage> CreatePipe<TMessage>(ChannelProviderBase parent, string name)
        {
            Contracts.AssertValue(parent);
            Contracts.Assert(parent is ConsoleEnvironment);
            Contracts.AssertNonEmpty(name);
            return new Pipe<TMessage>(parent, name, GetDispatchDelegate<TMessage>());
        }
    }
}

