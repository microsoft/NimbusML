//------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//------------------------------------------------------------------------------

using System;
using System.Globalization;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.MachineLearning.DotNetBridge
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

            public Host(HostEnvironmentBase<RmlEnvironment> source, string shortName, string parentFullName, IRandom rand, bool verbose, int? conc)
                : base(source, shortName, parentFullName, rand, verbose, conc)
            {
            }

            public new bool IsCancelled { get { return Root.IsCancelled; } }
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

            protected override IHost RegisterCore(HostEnvironmentBase<RmlEnvironment> source, string shortName, string parentFullName, IRandom rand, bool verbose, int? conc)
            {
                return new Host(source, shortName, parentFullName, rand, verbose, conc);
            }

        }

        public new bool IsCancelled { get { return CheckCancelled(); } }

        public RmlEnvironment(Bridge.CheckCancelled checkDelegate, int? seed = null, bool verbose = false, int conc = 0)
            : this(RandomUtils.Create(seed), verbose, conc)
        {

            CheckCancelled = checkDelegate;
        }

        public RmlEnvironment(IRandom rand, bool verbose = false, int conc = 0)
            : base(rand, verbose, conc)
        {
            CultureInfo.CurrentUICulture = CultureInfo.InvariantCulture;
            EnsureDispatcher<ChannelMessage>();
        }

        public RmlEnvironment(RmlEnvironment source, int? seed = null, bool verbose = false, int conc = 0)
            : this(source, RandomUtils.Create(seed), verbose, conc)
        {
        }

        public RmlEnvironment(RmlEnvironment source, IRandom rand, bool verbose = false, int conc = 0)
            : base(source, rand, verbose, conc)
        {
            CultureInfo.CurrentUICulture = CultureInfo.InvariantCulture;
            EnsureDispatcher<ChannelMessage>();
        }

        protected override IHost RegisterCore(HostEnvironmentBase<RmlEnvironment> source, string shortName, string parentFullName, IRandom rand, bool verbose, int? conc)
        {
            Contracts.AssertValue(rand);
            Contracts.AssertValueOrNull(parentFullName);
            Contracts.AssertNonEmpty(shortName);
            Contracts.Assert(source == this || source is Host);
            return new Host(source, shortName, parentFullName, rand, verbose, conc);
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

