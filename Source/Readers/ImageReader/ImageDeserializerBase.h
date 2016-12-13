//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "DataDeserializerBase.h"
#include "Config.h"
#include "CorpusDescriptor.h"
#include "ImageUtil.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    // Base class of image deserializers.
    class ImageDeserializerBase : public DataDeserializerBase
    {
    public:
        // A new constructor to support new compositional configuration,
        // that allows composition of deserializers and transforms on inputs.
        ImageDeserializerBase(CorpusDescriptorPtr corpus, const ConfigParameters& config);

        // Currently for backward compat with the old reader.
        ImageDeserializerBase();

    protected:
        void PopulateSequenceData(cv::Mat image, size_t classId, size_t sequenceId, std::vector<SequenceDataPtr>& result);

        // A helper class for generation of type specific labels (currently float/double only).
        LabelGeneratorPtr m_labelGenerator;

        // Mapping of logical sequence key into sequence description.
        std::map<size_t, size_t> m_keyToSequence;

        // Precision required by the network.
        ElementType m_precision;

        // Flag, whether images shall be loaded in grayscale.
        bool m_grayscale;

        // Verbosity.
        int m_verbosity;

        // Flag indicating whether to generate images for multi crop.
        bool m_multiViewCrop;

        CorpusDescriptorPtr m_corpus;
    };
}}}
