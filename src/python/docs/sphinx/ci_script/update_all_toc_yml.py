# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 10:11:10 2018

In this file, we refer to the markdown files for classes, such as
lightlda.md in folder modules/feature_extraction/text, as md class
reference file.
The corresponding nimbusml.feature_extraction.text.lightlda.yml in
docs-ref-autogen as yml class reference file.

In this script, we do the following things:
(1) Update the module.md/index.md to point to yml class reference file
instead of md class reference file
(2) Update all markdown files that point to md class reference file to
refer to to yml class reference file,
    such as in text.md in modules/feature_extraction, we update all the
    references in it to use yml files as reference
(3) In the yaml files, we have reference to classes, for example,
see also @logisticregression. However, those are rendered as
@../../nimbusml.feature_extraction.text.lightlda
    We should use @nimbusml.feature_extraction.text.lightlda no matter
    where this file is saved. We update all the class reference to
    @class_name
(4) There are two label reference, to refer to a section in other
markdown files, such as see @vector-type. They are rendered as @label_name.
    I am updating this to [label_name](file_path_to_md#label_name) so we
    can find the proper references.
(5) For a few markdown files, there is a table of content at the
beginning pointing to the sections below. They are rendered as [section
title](section_name.md).
    I am updating this to [section title](#section_name) because there
    is no markdownfiles named section_name.md.
"""
import os
import re
import shutil
import time

# 1 Update Module.md
print("Updating modules.md...")
file = r"_build\ms_doc_ref\nimbusml\modules.md"
file2 = r"_build\ms_doc_ref\nimbusml\modules2.md"
if not os.path.exists(os.path.dirname(file2)):
    os.makedirs(os.path.dirname(file2))

dict2 = {}
dict1 = {}
file_r = open(file, "r")
file_w = open(file2, "w")

write = 0
for line in file_r:

    if '*nimbusml' in line:
        class_name = re.search('(?<=\[\*)(.*)(?=\*\])', line)
        if not class_name:
            class_name = re.search('(?<=\[\*)(.*)(?=\*\:)', line).group(0)
        else:
            class_name = class_name.group(0)
        old_path = re.search('(?<=\()(.*)(?=\))', line).group(0)
        new_path = 'xref:' + class_name
        dict2[old_path] = new_path
        dict1[class_name] = new_path
        line = line.replace(old_path, new_path)

    if '(docs-ref-autogen/nimbusml.FileDataStream.yml)' in line:
        line = line.replace('(docs-ref-autogen/nimbusml.FileDataStream.yml)',
                            '(xref:nimbusml.FileDataStream)')
    file_w.write(line)

file_r.close()
file_w.close()
os.remove(r'_build\ms_doc_ref\nimbusml\modules.md')
os.rename(r'_build\ms_doc_ref\nimbusml\modules2.md',
          r'_build\ms_doc_ref\nimbusml\modules.md')
# 1.2 Update index.md

print("Updating index.md...")
file = r"_build\ms_doc_ref\nimbusml\index.md"
file2 = r"_build\ms_doc_ref\nimbusml\index2.md"
if not os.path.exists(os.path.dirname(file2)):
    os.makedirs(os.path.dirname(file2))

file_r = open(file, "r")
file_w = open(file2, "w")

write = 0
for line in file_r:

    if "Documentation" in line:
        break
    if '*nimbusml' in line:
        class_name = re.search('(?<=\[\*)(.*)(?=\*\])', line)
        if not class_name:
            class_name = re.search('(?<=\[\*)(.*)(?=\*\:)', line).group(0)
        else:
            class_name = class_name.group(0)
        old_path = re.search('(?<=\()(.*)(?=\))', line).group(0)
        new_path = 'xref:' + class_name
        line = line.replace(old_path, new_path)
    if '(docs-ref-autogen/nimbusml.FileDataStream.yml)' in line:
        line = line.replace(
            '(docs-ref-autogen/nimbusml.FileDataStream.yml)',
            '(xref:nimbusml.FileDataStream)')
    file_w.write(line)

file_r.close()
file_w.close()
os.remove(r'_build\ms_doc_ref\nimbusml\index.md')
os.rename(r'_build\ms_doc_ref\nimbusml\index2.md',
          r'_build\ms_doc_ref\nimbusml\index.md')
# 2 Update all md files
print("Updating md files under modules\\")
rootdir = r'_build\ms_doc_ref\nimbusml\modules'
for root, subdirs, files in os.walk(rootdir):
    for file_base in files:
        file = os.path.join(root, file_base)
        file2 = file.replace('modules', 'modules2')
        print(root)
        if not os.path.exists(os.path.dirname(file2)):
            os.makedirs(os.path.dirname(file2))

        try:
            file_r = open(file, "r")
            file_w = open(file2, "w")

            for line in file_r:

                if '[*nimbusml.' in line:
                    class_name = re.search('(?<=\[\*)(.*)(?=\*\])', line)
                    if not class_name:
                        class_name = re.search(
                            '(?<=\[\*)(.*)(?=\*\:)', line).group(0)
                    else:
                        class_name = class_name.group(0)
                    #                    print(class_name)
                    if class_name in dict1:
                        old_path = re.search(
                            '(?<=\()(.*)(?=\))', line).group(0)
                        if len(root.split('\\')) == 5:
                            new_path = 'xref:' + class_name
                        elif len(root.split('\\')) == 6:
                            new_path = 'xref:' + class_name
                        elif len(root.split('\\')) == 4:
                            new_path = 'xref:' + class_name
                        line = line.replace(old_path, new_path)
                    #                        print(new_path)
                    else:
                        print("Missing class: ")
                        print(file)
                        print(class_name)

                file_w.write(line)
        except Exception as e:
            print("not processed: " + str(file) + str(e))
        file_r.close()
        file_w.close()
file_r.close()
file_w.close()
shutil.rmtree(r'_build\ms_doc_ref\nimbusml\modules')
time.sleep(5)
os.rename(r'_build\ms_doc_ref\nimbusml\modules2',
          r'_build\ms_doc_ref\nimbusml\modules')

# 3 fix class reference####################
print("Updating all yaml files reference to fix class reference...")

rootdir = r'_build\ms_doc_ref\nimbusml\docs-ref-autogen'
for root, subdirs, files in os.walk(rootdir):
    for file_base in files:
        file = os.path.join(root, file_base)
        #        print(file)

        file2 = file.replace('docs-ref-autogen', 'docs-ref-autogen2')

        if not os.path.exists(os.path.dirname(file2)):
            os.makedirs(os.path.dirname(file2))

        try:
            file_r = open(file, "r")
            file_w = open(file2, "w")

            for line in file_r:
                if '@../../nimbusml.' in line:
                    # class_name = re.search('(?<=/nimbusml\.)(.*)(?=\ )',
                    # line).group(0)
                    line = line.replace('@../../nimbusml.', '@nimbusml.')
                if '@../../../nimbusml' in line:
                    line = line.replace('@../../../nimbusml', '@nimbusml')
                if '@../nimbusml' in line:
                    line = line.replace('@../nimbusml', '@nimbusml')
                if '@../../../../nimbusml' in line:
                    line = line.replace('@../../../../nimbusml', '@nimbusml')
                file_w.write(line)

        except Exception as e:
            print("not processed: " + str(file) + ',' + str(e))
        file_r.close()
        file_w.close()
file_r.close()
file_w.close()
shutil.rmtree(r'_build\ms_doc_ref\nimbusml\docs-ref-autogen')

time.sleep(5)
os.rename(r'_build\ms_doc_ref\nimbusml\docs-ref-autogen2',
          r'_build\ms_doc_ref\nimbusml\docs-ref-autogen')

# 4 fix label reference
print("Updating all yaml files reference to fix label reference...")
rootdir = r'_build\ms_doc_ref\nimbusml\docs-ref-autogen'

for root, subdirs, files in os.walk(rootdir):
    for file_base in files:
        file = os.path.join(root, file_base)
        file2 = file.replace('docs-ref-autogen', 'docs-ref-autogen2')

        if not os.path.exists(os.path.dirname(file2)):
            os.makedirs(os.path.dirname(file2))

        try:
            file_r = open(file, "r")
            file_w = open(file2, "w")

            for line in file_r:

                if '@l-pipeline-syntax' in line:
                    line = line.replace(
                        '@l-pipeline-syntax',
                        '[Columns]('
                        '../concepts/columns.md#l-pipeline-syntax)')
                if '@vectortype' in line:
                    line = line.replace(
                        '@vectortype',
                        '[Vector Type]('
                        '../concepts/types.md#vectortype-columns)')
                if '@column-types' in line:
                    line = line.replace(
                        '@column-types',
                        '[Types](../concepts/types.md#column-types)')
                if '*column-types*' in line:
                    line = line.replace(
                        '*column-types*',
                        '[Types](../concepts/types.md#column-types)')
                if '@loss_intro' in line:
                    line = line.replace(
                        '@loss_intro',
                        '[Loss Intro](../modules/loss.md#loss_intro)')
                if '*loss_intro*' in line:
                    line = line.replace(
                        '*loss_intro*',
                        '[Loss Intro](../modules/loss.md#loss_intro)')
                if '@loss-functions' in line:
                    line = line.replace(
                        '@loss-functions',
                        '[API Guide: Loss Functions]('
                        '../apiguide.md#loss-functions)')
                if '@columntypes' in line:
                    line = line.replace(
                        '@columntypes',
                        '[Schema](../concepts/schema.md#dataschema-class)')
                if '@roles' in line:
                    line = line.replace(
                        '@roles',
                        '[Roles](../concepts/roles.md#roles-and-learners)')
                if '../modules/Pipeline.md#pipeline' in line:
                    line = line.replace(
                        '../modules/Pipeline.md#pipeline',
                        '../docs-ref-autogen/nimbusml.Pipeline.yml)')
                if '../modules/data/DataSchema.md#dataschema' in line:
                    line = line.replace(
                        '../modules/data/DataSchema.md#dataschema',
                        '../docs-ref-autogen/nimbusml.DataSchema.yml)')
                if '../modules/data/Role.md#role' in line:
                    line = line.replace(
                        '../modules/data/Role.md#role',
                        '../docs-ref-autogen/nimbusml.Role.yml)')
                if '../modules/data/FileDataStream.md#filedatastream' in \
                        line:
                    line = line.replace(
                        '../modules/data/FileDataStream.md#filedatastream',
                        '../docs-ref-autogen/nimbusml.FileDataStream.yml)')
                file_w.write(line)
        except Exception as e:
            print("not processed: " + str(file) + ',' + str(e))
        file_r.close()
        file_w.close()
shutil.rmtree(r'_build\ms_doc_ref\nimbusml\docs-ref-autogen')
time.sleep(5)
os.rename(r'_build\ms_doc_ref\nimbusml\docs-ref-autogen2',
          r'_build\ms_doc_ref\nimbusml\docs-ref-autogen')

# 5 fix section reference in md files, in the same md files
print("Updating all md files reference to fix local section reference...")
rootdir = r'_build\ms_doc_ref\nimbusml'

for root, subdirs, files in os.walk(rootdir):
    for file_base in files:
        if 'concepts' in root or file_base in ['apiguide.md',
                                               'installationguide.md',
                                               'index.md', 'metrics.md',
                                               'loadsavemodels.md', 'concepts.md']:

            file = os.path.join(root, file_base)

            file2 = file.replace('.', '2.')

            if not os.path.exists(os.path.dirname(file2)):
                os.makedirs(os.path.dirname(file2))
            writeline = 1
            if '.md' in file:
                try:
                    file_r = open(file, "r")
                    file_w = open(file2, "w")

                    for line in file_r:
                        file_name = re.search('(?<=\()(.*)(?=.md\))', line)
                        if file_name:
                            file_name_full = file_name.group(
                                0).split('(')[-1] + '.md'
                            if not os.path.exists(
                                    os.path.join(
                                        root,
                                        file_name_full)) and '/' not in \
                                    file_name_full:
                                print('look for file: ' + file_name_full)
                                print('processing file: ' + file)
                                line = line.replace(
                                    file_name_full,
                                    '#' + file_name.group(0).split('(')[
                                        -1])

                        image_file_name = re.search(
                            '(?<=\!\[alt\]\()(.*)(?=\.png)', line)
                        if image_file_name:
                            image_file_name = image_file_name.group(0)
                            line = line.replace('alt', 'png')
                            line = line.replace(' png)', ')')
                            if 'car' in image_file_name:
                                line = line.replace('_images',
                                                    '../_images')
                        if '../modules/Pipeline.md#pipeline' in line:
                            line = line.replace(
                                '../modules/Pipeline.md#pipeline',
                                'xref:nimbusml.Pipeline')
                        if '../modules/data/DataSchema.md#dataschema' in \
                                line:
                            line = line.replace(
                                '../modules/data/DataSchema.md#dataschema',
                                'xref:nimbusml.DataSchema')
                        if '../modules/data/Role.md#role' in \
                                line:
                            line = line.replace(
                                '../modules/data/Role.md#role',
                                'xref:nimbusml.Role')
                        if '../modules/data/FileDataStream.md#filedatastream' \
                                in line:
                            line = line.replace(
                                '../modules/data/FileDataStream.md#'
                                'filedatastream',
                                'xref:nimbusml.FileDataStream')
                        if 'modules/Pipeline.md#pipeline' in line:
                            line = line.replace(
                                'modules/Pipeline.md#pipeline',
                                'xref:nimbusml.Pipeline'
                            )
                        if "#tutorials" in line:
                            line = line.replace(
                                "#tutorials", "tutorials.md#tutorials")
                        if "[sklearn.Pipeline]" in line:
                            line = line.replace(
                                "[sklearn.Pipeline]",
                                "[`sklearn.Pipeline`]")
                        if "and-columns-are-interchangeable" in line:
                            line = line.replace(
                                "and-columns-are-interchangeable",
                                "-and-columns-are-interchangeable")
                        if "use-operator-to-select-columns" in line:
                            line = line.replace(
                                "use-operator-to-select-columns",
                                "use--operator-to-select-columns")
                        if "[Column Roles for Trainers](roles.md#roles)" in line:
                            line = line.replace(
                                "[Column Roles for Trainers](roles.md#roles)",
                                "[Column Roles for Trainers](roles.md#roles-and-learners)")
                        if "[VectorDataViewType Columns](types.md#vectortype)" in line:
                            line = line.replace(
                                "[VectorDataViewType Columns](types.md#vectortype)",
                                "[VectorDataViewType Columns](types.md#vectortype-columns)")
                        if "[Column Operations for Transforms](columns.md#l-pipeline-syntax)" in line:
                            line = line.replace(
                                "[Column Operations for Transforms](columns.md#l-pipeline-syntax)",
                                "[Column Operations for Transforms](columns.md#how-to-select-columns-to-transform)")
                        if "[Schema](schema.md#schema)" in line:
                            line = line.replace(
                                "[Schema](schema.md#schema)",
                                "[Schema](schema.md)")
                        if "[Data Sources](datasources.md#datasources)" in line:
                            line = line.replace(
                                "[Data Sources](datasources.md#datasources)",
                                "[Data Sources](datasources.md)")
                        if "[tutorial section](tutorials.md#tutorials)" in line:
                            line = line.replace(
                                "[tutorial section](tutorials.md#tutorials)",
                                "[tutorial section](tutorials.md)")
                        if "scoring-in-ml-net" in line:
                            line = line.replace(
                                "scoring-in-ml-net",
                                "scoring-in-nimbusml")
                        if "# Description" in line:
                            writeline = -6

                        if "supported_version" in line:
                            version_table = \
                                """| Python Version \ Platform | Windows | Linux |  Mac  |
|:-------------------------:|:-------:|:-----:|:-----:|
|            2.7            |   Yes   |  Yes  |  Yes  |
|            3.5            |   Yes   |  Yes  |  Yes  |
|            3.6            |   Yes   |  Yes  |  Yes  |
|            3.7            |   Yes   |  Yes  |  Yes  |"""
                            file_w.write(version_table)
                            writeline = 0
                        if "../modules/data/FileDataStream.md#readcsv" in line:
                            line = line.replace(
                                "../modules/data/FileDataStream.md#readcsv",
                                'xref:nimbusml.FileDataStream.read_csv'
                            )
                        if "load_save_model_csharp" in line:
                            csharp_code = \
                                """```csharp
public async void Score()
{
    var modelName = "mymodeluci.zip";
    var pipeline = new Legacy.LearningPipeline();
    var loadedModel = await Legacy.PredictionModel.ReadAsync<InfertData, InfertPrediction>(modelName);
    var singlePrediction = loadedModel.Predict(new InfertData()
    {
        age = 26,
        parity = 6,
        spontaneous = 2,
    });

    Console.WriteLine(singlePrediction.ToString());
}
public class InfertData
{
    [Column(ordinal: "0")]
    public int age;

    [Column(ordinal: "1")]
    public int parity;

    [Column(ordinal: "2")]
    public int spontaneous;
}

public class InfertPrediction
{
    [ColumnName("PredictedLabel")]
    public bool PredictedLabel;
    [ColumnName("Probability")]
    public float Probability;
    [ColumnName("Score")]
    public float Score;
}
```"""
                            file_w.write(csharp_code)
                            writeline = 0
                        if writeline > 0:
                            file_w.write(line)
                        writeline += 1

                    file_r.close()
                    file_w.close()
                    os.remove(file)
                    os.rename(file2, file)
                except Exception as e:
                    print("not processed: " + str(e) + str(file))
                file_r.close()
                file_w.close()


# Fix ms-scikit.md. Link to yml files instead of md files


def createlinktoyaml(m):
    cname = m.groups()[0]
    cpath = m.groups()[1]
    cpath = cpath.replace('/', '.')
    cpath = cpath.replace('-', '_')
    cpath = cpath.replace('modules', r'docs-ref-autogen/nimbusml')
    cpath = cpath.replace('.md', '.yml')
    return '[{}]({})'.format(cname, cpath)


files = [
    os.path.join(
        os.path.normpath(__file__),
        '..',
        '..',
        '_build',
        'ms_doc_ref',
        'nimbusml',
        'ms-scikit.md')]
for file in files:
    print("processing file: ", file)
    with open(file, 'r') as infile:
        lines = infile.readlines()
    with open(file, 'w') as outfile:
        for line in lines:
            p = re.compile(r'\[(.*?)\]\((.*?)\)')
            line = p.sub(createlinktoyaml, line)
            outfile.write(line)
