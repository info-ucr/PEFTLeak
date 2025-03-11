import torch
from k_means_constrained import KMeansConstrained

def match_cluster( elements, total_cluster, num_patch):
        #"""Match (or rather cluster) the sentencen components into at maximum shape[0] sentences of length shape[1]"""
        #if algorithm == "k-means":
        

        clustering = KMeansConstrained(
                n_clusters=total_cluster,
                size_min=0, #num_patch,
                size_max=min(num_patch, len(elements)),
                init="k-means++",
                n_init=40,
                max_iter=900,
                tol=1e-6,
            )
        
        #elements=elements.reshape(768)
        #print(len(elements[0]))
        std, mean = torch.std_mean(elements, dim=-1, keepdim=True)
        # print(len(std))
        # print(len(mean))
        normalized_components = (elements - mean) / (std + 1e-10)
        
        normalized_components=elements
        #print(normalized_components)
        labels = clustering.fit_predict(normalized_components.double().numpy())
        labels = torch.as_tensor(labels, dtype=torch.long)
        return labels