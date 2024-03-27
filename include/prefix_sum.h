// prefix_sum.h

template< typename InputIt, typename OutputIt >
void prefix_sum(InputIt b, const InputIt e, OutputIt out){
  std::partial_sum(b, e, out);
} 
