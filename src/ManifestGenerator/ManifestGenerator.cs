//------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//------------------------------------------------------------------------------

using System;
using Microsoft.ML.DotNetBridge;


namespace Microsoft.ML.ManifestGenerator
{
    public static class ManifestGenerator
    {
        public static void ShowUsage()
        {
            string usage =
            "Usage:\n" +
            "    create MANIFEST_PATH        Creates a new manifest given the\n" +
            "                                current assemblies and stores it\n" +
            "                                in the file MANIFEST_PATH.\n" +
            "\n";

            Console.WriteLine(usage);
        }

        public static void Main(string[] args)
        {
            bool argsValid = false;

            if (args.Length == 2)
            {
                if (args[0].ToLower() == "create")
                {
                    argsValid = true;

                    ManifestUtils.ShowAssemblyInfo();
                    ManifestUtils.GenerateManifest(args[1]);
                }
            }

            if (!argsValid)
            {
                Console.WriteLine("ManifestGenerator: Error - Invalid Arguments.");
                ShowUsage();
            }
        }
    }
}
