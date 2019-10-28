//------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//------------------------------------------------------------------------------

using System;
using System.IO;
using System.Linq;
using Microsoft.ML.DotNetBridge;


namespace Microsoft.ML.ManifestGenerator
{
    public static class ManifestGenerator
    {
        private const int ERROR_SUCCESS = 0;
        private const int ERROR_BAD_ARGUMENTS = 1;
        private const int ERROR_MANIFEST_INVALID = 2;

        public static void ShowUsage()
        {
            string usage =
            "Usage:\n" +
            "    create MANIFEST_PATH        Creates a new manifest given the\n" +
            "                                current assemblies and stores it\n" +
            "                                in the file MANIFEST_PATH.\n" +
            "    verify MANIFEST_PATH        Checks if the manifest specified by\n" +
            "                                MANIFEST_PATH is valid given the\n" +
            "                                the current assemblies.\n" +
            "\n";

            Console.WriteLine(usage);
        }

        public static int Main(string[] args)
        {
            int exitCode = ERROR_BAD_ARGUMENTS;

            if (args.Length == 2)
            {
                if (args[0].ToLower() == "create")
                {
                    ManifestUtils.ShowAssemblyInfo();
                    ManifestUtils.GenerateManifest(args[1]);

                    exitCode = ERROR_SUCCESS;
                }
                else if (args[0].ToLower() == "verify")
                {
                    string tmpFilePath = Path.GetTempFileName();
                    ManifestUtils.GenerateManifest(tmpFilePath);

                    exitCode = FilesMatch(args[1], tmpFilePath) ?
                               exitCode = ERROR_SUCCESS :
                               exitCode = ERROR_MANIFEST_INVALID;

                    File.Delete(tmpFilePath);
                }
            }

            if (exitCode == ERROR_BAD_ARGUMENTS)
            {
                Console.WriteLine("ManifestGenerator: Error - Invalid Arguments.");
                ShowUsage();
            }

            return exitCode;
        }

        private static bool FilesMatch(string path1, string path2)
        {
            long fileLength1 = new FileInfo(path1).Length;
            long fileLength2 = new FileInfo(path2).Length;
            if (fileLength1 != fileLength2) return false;

            // TODO: read in only parts of the file at a time
            bool bytesMatch = File.ReadAllBytes(path1).SequenceEqual(File.ReadAllBytes(path2));
            return bytesMatch;
        }
    }
}
